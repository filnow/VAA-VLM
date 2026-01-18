import os
import io
import csv
import sys
import json
import base64
import logging
import torch
import wandb
import hydra
import random
import numpy as np
import pandas as pd
from openai import OpenAI
from omegaconf import DictConfig, OmegaConf

from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from typing import List, Dict, Optional, Set, Union, Any

logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
log = logging.getLogger(__name__)
load_dotenv()

NestedRatings = Dict[str, Dict[str, Dict[str, Union[float, str]]]]

def set_all_seeds(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"All random seeds set to: {seed}")

def encode_image(pil_image: Image.Image) -> str:
    """Encode a PIL image to a base64 string."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def calculate_summary_stats(ratings_df: pd.DataFrame) -> dict:
    """
    Calculates overall summary statistics from the regression ratings DataFrame.
    """
    stats = {}
    # Find all columns that contain mean ratings and standard deviations
    mean_cols = [col for col in ratings_df.columns if col.endswith('_Mean')]
    std_cols = [col for col in ratings_df.columns if col.endswith('_Std')]

    if not mean_cols:
        return stats # Return empty if no data

    # Calculate the mean of all the mean ratings (a single number summarizing the run)
    stats['grand_mean_rating'] = ratings_df[mean_cols].values.mean()
    
    # Calculate the mean of all the standard deviations (summarizes rating consistency)
    if std_cols:
        stats['avg_rating_std'] = ratings_df[std_cols].values.mean()
        
    log.info(f"Calculated Summary Stats: {stats}")
    return stats

class DataHandler:
    """Handles all data loading operations."""
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def load_dataset(self):
        log.info(f"Loading dataset from Hugging Face: {self.cfg.dataset.hf_path}")
        return load_dataset(self.cfg.dataset.hf_path)["train"]

    def load_emotion_definitions(self) -> dict:
        log.info(f"Loading emotion definitions from: {self.cfg.emotion_definitions_path}")
        try:
            with open(self.cfg.emotion_definitions_path, 'r', encoding='utf-8') as f:
                all_definitions = json.load(f)
            return all_definitions.get(self.cfg.dataset.name, {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            log.error(f"Failed to load emotion definitions: {e}")
            raise

class ModelEvaluator:
    """Handles model initialization, prompt creation, and emotion rating."""
    SPECIAL_EMOTION_ALIASES: Dict[str, List[str]] = {
        "motivatedtoapproach": ["Approach"],
        "motivatedtoavoid": ["Avoid"],
    }
    def __init__(self, cfg: DictConfig, valid_image_filenames: Optional[Set[str]] = None):
        self.cfg = cfg
        self.model_cfg = cfg.model
        self.sampling_cfg = cfg.sampling_params
        
        model_name = self.model_cfg.name.lower()
        if "gemini" not in model_name:
            raise ValueError("This pipeline is configured for Gemini models only.")

        log.info(f"Initializing Gemini model: {self.model_cfg.name}")
        self.llm = OpenAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        self.prompt_templates = self._load_prompt_templates(cfg.prompt_config_path)
        self.context_data = self._load_context_for_image(valid_image_filenames)

    def _load_prompt_templates(self, path: str) -> dict:
        log.info(f"Loading prompt templates from: {path}")
        with open(path, 'r') as f:
            return json.load(f)
        
    def _load_context_for_image(self, valid_filenames: Optional[Set[str]] = None) -> dict:
        with open("./data/human_context/laigai-human-context.json", 'r') as f:
            context_data = json.load(f)

        if not valid_filenames:
            return context_data

        normalized_valid = {str(name).strip() for name in valid_filenames if name}
        filtered_context = {name: context_data[name] for name in normalized_valid if name in context_data}

        missing_context = normalized_valid - set(filtered_context.keys())
        if missing_context:
            log.warning(
                "Context not found for %d image(s) present in the filtered dataset."
                " They'll receive empty context when prompting.",
                len(missing_context)
            )

        log.info(
            "Loaded context for %d/%d filtered images (%.2f%% coverage).",
            len(filtered_context),
            len(normalized_valid),
            (len(filtered_context) / len(normalized_valid) * 100) if normalized_valid else 0.0
        )

        return filtered_context


    def _create_batch_prompt_for_image(
        self,
        base64_image: str,
        emotion_definitions: dict,
        context: Optional[dict] = None
    ) -> List[Dict]:
        context = context or {}
        scale = self.cfg.dataset.scale
        system_content = self.prompt_templates['system_content_template'].format(scale_of_emotion=scale)
        assistant_content = self.prompt_templates['assistant_content_template'].format(scale_of_emotion=scale)

        user_sections = []
        template = self.prompt_templates['user_text_template']
        safe_template = template.replace("{_Before_{emotion_name}}", "{before_value}")

        for emotion_name, definition in emotion_definitions.items():
            before_value = context.get(
                f"_Before_{emotion_name}",
                context.get(f"{emotion_name}_Before", "unknown")
            )

            combined_context = {
                **context,
                "emotion_name": emotion_name,
                "scale_of_emotion": scale,
                "definition": definition,
                "before_value": before_value
            }

            user_sections.append(safe_template.format(**combined_context))

        combined_user_text = "###\n".join(user_sections)
        
        return [
            {"role": "system", "content": system_content},
            {"role": "assistant", "content": assistant_content},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": combined_user_text}
                ]
            }
        ]

    @staticmethod
    def _parse_emotion_scores(raw_content: Optional[str], expected_count: int) -> Optional[List[float]]:
        if not raw_content:
            return None

        tokens = [token.strip() for token in raw_content.replace("\n", ",").split(",")]
        scores: List[float] = []
        for token in tokens:
            if not token:
                continue
            try:
                scores.append(float(token))
            except ValueError:
                log.debug("Failed to parse score token '%s'", token)
                return None

        if len(scores) != expected_count:
            log.debug(
                "Parsed %d scores but expected %d based on emotion definitions.",
                len(scores),
                expected_count
            )
            return None

        return scores

    @staticmethod
    def _normalize_emotion_label(label: str) -> str:
        return "".join(ch for ch in label.lower() if ch.isalnum())

    def _lookup_before_value(self, participant_context: dict, emotion_name: str) -> Optional[Any]:
        target_norm = self._normalize_emotion_label(emotion_name)

        for alias in self.SPECIAL_EMOTION_ALIASES.get(target_norm, []):
            alias_key = f"{alias}_Before"
            if alias_key in participant_context:
                return participant_context[alias_key]

            alt_alias_key = f"{alias.replace(' ', '_')}_Before"
            if alt_alias_key in participant_context:
                return participant_context[alt_alias_key]

        direct_key = f"{emotion_name}_Before"
        if direct_key in participant_context:
            return participant_context[direct_key]

        alt_direct_key = f"{emotion_name.replace(' ', '_')}_Before"
        if alt_direct_key in participant_context:
            return participant_context[alt_direct_key]

        for key, value in participant_context.items():
            if not key.lower().endswith("_before"):
                continue
            base_label = key[:-7]
            if self._normalize_emotion_label(base_label) == target_norm:
                return value

        return None

    def _prepare_prompt_context(self, participant_context: dict, emotion_definitions: dict) -> dict:
        base_context = {
            "age": participant_context.get("age", "unknown"),
            "gender": participant_context.get("gender", "unknown"),
            "country": participant_context.get("country", "unknown"),
        }
        # Include all provided participant metadata for template flexibility
        base_context.update(participant_context)

        for emotion_name in emotion_definitions.keys():
            placeholder_key = f"_Before_{emotion_name}"
            if placeholder_key not in base_context:
                before_value = self._lookup_before_value(participant_context, emotion_name)
                base_context[placeholder_key] = before_value if before_value is not None else "unknown"
        return base_context

    def rate_emotions(
        self,
        dataset,
        emotion_definitions: dict
    ) -> NestedRatings:
        """Runs inference on the dataset, generating emotion ratings."""
        results: NestedRatings = {}
        emotions_to_rate = list(emotion_definitions.keys())
        expected_count = len(emotions_to_rate)

        for item in tqdm(dataset, desc="Processing images"):
            filename = str(item.get('filename', '')).strip()
            if not filename:
                log.warning("Encountered dataset item without a filename; skipping entry.")
                continue

            base64_image = encode_image(item['image'])
            participant_contexts = self.context_data.get(filename, {})
            if not participant_contexts:
                log.warning("No participant context found for image %s; skipping participant-level prompts.", filename)
                results[filename] = {}
                continue

            results[filename] = {}

            for participant_id, participant_context in participant_contexts.items():
                participant_id = str(participant_id)
                prompt_context = self._prepare_prompt_context(participant_context, emotion_definitions)
                batch_prompt = self._create_batch_prompt_for_image(
                    base64_image,
                    emotion_definitions,
                    prompt_context
                )

                participant_results: Dict[str, Union[float, str]] = {}

                try:
                    outputs = self.llm.chat.completions.create(
                        model=self.model_cfg.name,
                        n=1,
                        messages=batch_prompt,
                        max_tokens=self.sampling_cfg.max_tokens,
                        temperature=self.sampling_cfg.temperature,
                        reasoning_effort="none"
                    )
                except Exception as e:
                    log.error(
                        "API call failed for image %s (participant %s): %s",
                        filename,
                        participant_id,
                        str(e)
                    )
                    for emotion_name in emotions_to_rate:
                        participant_results[emotion_name] = "BLOCKED"
                    results[filename][participant_id] = participant_results
                    continue

                if not outputs.choices or not outputs.choices[0].message:
                    log.warning(
                        "No valid response received for image %s (participant %s).",
                        filename,
                        participant_id
                    )
                    for emotion_name in emotions_to_rate:
                        participant_results[emotion_name] = "BLOCKED"
                    results[filename][participant_id] = participant_results
                    continue

                raw_content = outputs.choices[0].message.content
                if not raw_content or raw_content.strip().lower() == "i'm sorry, but i can't assist with that request":
                    log.info(
                        "Model refused request for image %s (participant %s).",
                        filename,
                        participant_id
                    )
                    for emotion_name in emotions_to_rate:
                        participant_results[emotion_name] = "BLOCKED"
                    results[filename][participant_id] = participant_results
                    continue

                parsed_scores = self._parse_emotion_scores(raw_content, expected_count)
                if not parsed_scores:
                    log.warning(
                        "Unable to parse scores for image %s (participant %s). Raw response: %s",
                        filename,
                        participant_id,
                        raw_content
                    )
                    for emotion_name in emotions_to_rate:
                        participant_results[emotion_name] = "BLOCKED"
                else:
                    for emotion_name, score in zip(emotions_to_rate, parsed_scores):
                        participant_results[emotion_name] = round(score, 2)

                results[filename][participant_id] = participant_results

        return results

class ResultLogger:
    """Handles saving results to disk and logging to Weights & Biases."""
    def __init__(self, cfg: DictConfig):
        self.output_cfg = cfg.output
        self.wandb_cfg = cfg.wandb

    @staticmethod
    def _ratings_dict_to_dataframe(
        ratings: NestedRatings
    ) -> pd.DataFrame:
        records = []
        for image_name, participants in ratings.items():
            for participant_id, emotions in participants.items():
                row = {
                    "image": image_name,
                    "participant": participant_id
                }
                for emotion_name, values in emotions.items():
                    row[emotion_name] = values
                records.append(row)
        return pd.DataFrame.from_records(records)

    def save_to_csv(
        self,
        ratings: Union[pd.DataFrame, NestedRatings]
    ) -> str:
        """Saves the emotion ratings to a CSV file."""
        if isinstance(ratings, pd.DataFrame):
            df = ratings
        else:
            df = self._ratings_dict_to_dataframe(ratings)

        output_path = self.output_cfg.filename
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
        log.info(f"Results successfully saved to: {os.path.join(os.getcwd(), output_path)}")
        return output_path

    def save_to_json(self, data: Dict[str, Any]) -> str:
        """Saves the nested emotion ratings to a JSON file."""
        base_output = self.output_cfg.filename
        json_output_path = os.path.splitext(base_output)[0] + ".json"
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.info(f"Results successfully saved to JSON: {os.path.join(os.getcwd(), json_output_path)}")
        return json_output_path

    def log_to_wandb(
        self,
        stats: dict,
        ratings_data: NestedRatings,
        output_path: str,
        full_config: DictConfig
    ):
        """
        Initializes a W&B run and logs summary stats, the CSV as an artifact,
        and the data as a W&B Table.
        """
        if not self.wandb_cfg:
            log.info("W&B config not found, skipping logging.")
            return

        run_name = self.wandb_cfg.run_name
        project = self.wandb_cfg.project
        log.info(f"Initializing W&B run '{run_name}' in project '{project}'")
        
        notes = f"Uploading regression results for {full_config.dataset.name} with model {full_config.model.name}"

        with wandb.init(
            project=project,
            name=run_name,
            job_type=self.wandb_cfg.job_type,
            config=OmegaConf.to_container(full_config, resolve=True),
            notes=notes
        ) as run:
            
            # 1. Log summary statistics
            if stats:
                run.log(stats)
                log.info("Logged summary stats to W&B.")

            # 2. Log results CSV as a W&B Artifact for versioning
            log.info("Logging results CSV as a W&B Artifact...")
            artifact = wandb.Artifact(name=f'{run.name}-ratings', type='dataset')
            artifact.add_file(output_path)
            run.log_artifact(artifact)
            log.info("Artifact logging complete.")

            # 3. Log data as a W&B Table for visualization
            log.info("Logging data as a W&B Table...")
            ratings_df = self._ratings_dict_to_dataframe(ratings_data)
            table = wandb.Table(dataframe=ratings_df)
            run.log({f"{full_config.dataset.name}_ratings_table": table})
            log.info("Table logging complete.")


@hydra.main(version_base=None, config_path="configs", config_name="config-context-batch")
def main(cfg: DictConfig) -> None:
    """Main function to run the end-to-end emotion regression pipeline."""
    log.info("Starting regression run...")
    log.info("Full configuration for this run:\n" + OmegaConf.to_yaml(cfg))

    set_all_seeds(cfg.seed)
    # --- 1. Load Data ---
    data_handler = DataHandler(cfg)
    dataset = data_handler.load_dataset().select(range(80))
    emotion_definitions = data_handler.load_emotion_definitions()
    dataset_filenames = {str(name).strip() for name in dataset["filename"] if name}

    # --- 2. Run Model Evaluation ---
    evaluator = ModelEvaluator(cfg, valid_image_filenames=dataset_filenames)
    ratings_data = evaluator.rate_emotions(dataset, emotion_definitions)

    # --- 3. Calculate Stats, Save, and Log Results ---
    logger = ResultLogger(cfg)
    output_json_path = logger.save_to_json(ratings_data)
    #logger.log_to_wandb(summary_stats, ratings_data, output_json_path, cfg)

    log.info("Script finished successfully.")

if __name__ == "__main__":
    main()