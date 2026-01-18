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
from typing import List, Dict

logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
log = logging.getLogger(__name__)
load_dotenv()

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
        return load_dataset(self.cfg.dataset.hf_path)['train']

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
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model_cfg = cfg.model
        self.sampling_cfg = cfg.sampling_params
        
        if "gpt" in self.model_cfg.name:
            log.info(f"Initializing OpenAI model: {self.model_cfg.name}")
            self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif "gemini" in self.model_cfg.name:
            log.info(f"Initializing Gemini model: {self.model_cfg.name}")
            self.llm = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"),
                              base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                            )
        else:
            log.info("None of the specified models matched.")

        self.prompt_templates = self._load_prompt_templates(cfg.prompt_config_path)

    def _load_prompt_templates(self, path: str) -> dict:
        log.info(f"Loading prompt templates from: {path}")
        with open(path, 'r') as f:
            return json.load(f)

    def _create_batch_prompt_for_image(self, base64_image: str, emotion_definitions: dict) -> List[Dict]:
        scale = self.cfg.dataset.scale
        system_content = self.prompt_templates['system_content_template'].format(scale_of_emotion=scale)
        assistant_content = self.prompt_templates['assistant_content_template'].format(scale_of_emotion=scale)

        user_sections = []
        for emotion_name, definition in emotion_definitions.items():
            user_sections.append(
                self.prompt_templates['user_text_template'].format(
                    emotion_name=emotion_name,
                    scale_of_emotion=scale,
                    definition=definition
                )
            )

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

    def rate_emotions(self, dataset, emotion_definitions: dict) -> pd.DataFrame:
        """Runs inference on the dataset, generating emotion ratings."""
        results = []
        emotions_to_rate = list(emotion_definitions.keys())
        predicted_text = ""

        for item in tqdm(dataset, desc="Processing images"):
            base64_image = encode_image(item['image'])
            batch_prompt = self._create_batch_prompt_for_image(base64_image, emotion_definitions)
            scores_from_batch = []
            predicted_text = "OK"

            for i in range(10):
                try:
                    if "gpt" in self.model_cfg.name:
                        outputs = self.llm.chat.completions.create(
                            model=self.model_cfg.name,
                            n=6,
                            messages=batch_prompt,
                            max_tokens=50,
                            temperature=self.sampling_cfg.temperature,
                            seed=self.sampling_cfg.seed,
                        )
                    elif "gemini" in self.model_cfg.name:
                        outputs = self.llm.chat.completions.create(
                            model=self.model_cfg.name,
                            n=6,
                            messages=batch_prompt,
                            max_tokens=50,
                            temperature=self.sampling_cfg.temperature,
                            reasoning_effort="none"
                        )
                    if (outputs.choices[0].message.content is None) or \
                        (outputs.choices[0].message.content.strip().lower() == "i'm sorry, but i can't assist with that request"):
                        predicted_text = "BLOCKED"
                        break
                    
                    for i in range(1, len(outputs.choices)):
                        scores = []
                        for x in outputs.choices[i].message.content.split(","):
                            x = x.strip()
                            if x.isdigit():
                                scores.append(float(x))
                        scores_from_batch.append(scores)
                
                except Exception as e:
                    log.error(f"API call failed for image {item['filename']}: {str(e)}")
                    predicted_text = "BLOCKED"
                    break

            row_data = {'image': item['filename']}
            if predicted_text == "BLOCKED":
                for emotion_name in emotions_to_rate:
                    row_data[f"{emotion_name}_Mean"] = "BLOCKED"
                    row_data[f"{emotion_name}_Std"] = "BLOCKED"
            else:
                num_emotions = len(scores_from_batch[0]) if scores_from_batch else 0
                
                for emotion_idx in range(num_emotions):
                    emotion_name = emotions_to_rate[emotion_idx]
                    scores_at_position = [scores[emotion_idx] for scores in scores_from_batch if emotion_idx < len(scores)]
                    
                    if not scores_at_position:
                        mean_score, std_score = float('nan'), float('nan')
                    else:
                        mean_score = round(sum(scores_at_position) / len(scores_at_position), 2)
                        std_score = round(torch.std(torch.tensor(scores_at_position)).item(), 2) if len(scores_at_position) > 1 else 0.0
                    row_data[f"{emotion_name}_Mean"] = mean_score
                    row_data[f"{emotion_name}_Std"] = std_score

            results.append(row_data)
        return pd.DataFrame(results)

class ResultLogger:
    """Handles saving results to CSV and logging to Weights & Biases."""
    def __init__(self, cfg: DictConfig):
        self.output_cfg = cfg.output
        self.wandb_cfg = cfg.wandb

    def save_to_csv(self, df: pd.DataFrame) -> str:
        """Saves the DataFrame to a CSV file."""
        output_path = self.output_cfg.filename
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
        log.info(f"Results successfully saved to: {os.path.join(os.getcwd(), output_path)}")
        return output_path

    def log_to_wandb(self, stats: dict, ratings_df: pd.DataFrame, output_path: str, full_config: DictConfig):
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
            table = wandb.Table(dataframe=ratings_df)
            run.log({f"{full_config.dataset.name}_ratings_table": table})
            log.info("Table logging complete.")


@hydra.main(version_base=None, config_path="configs", config_name="config-reg-batch")
def main(cfg: DictConfig) -> None:
    """Main function to run the end-to-end emotion regression pipeline."""
    log.info("Starting regression run...")
    log.info("Full configuration for this run:\n" + OmegaConf.to_yaml(cfg))

    set_all_seeds(cfg.seed)
    # --- 1. Load Data ---
    data_handler = DataHandler(cfg)
    dataset = data_handler.load_dataset()
    emotion_definitions = data_handler.load_emotion_definitions()

    # --- 2. Run Model Evaluation ---
    evaluator = ModelEvaluator(cfg)
    ratings_df = evaluator.rate_emotions(dataset, emotion_definitions)

    # --- 3. Calculate Stats, Save, and Log Results ---
    # summary_stats = calculate_summary_stats(ratings_df)
    logger = ResultLogger(cfg)
    output_csv_path = logger.save_to_csv(ratings_df)
    #logger.log_to_wandb(summary_stats, ratings_df, output_csv_path, cfg)

    log.info("Script finished successfully.")

if __name__ == "__main__":
    main()
