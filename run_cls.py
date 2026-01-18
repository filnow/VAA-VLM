import os
import io
import csv
import json
import base64
import logging
import pandas as pd
import wandb
import hydra
import random
import numpy as np
import torch

from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import classification_report


logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
log = logging.getLogger(__name__)

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

def calculate_metrics(predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> dict:
    """
    Calculate overall and per-emotion metrics, returning a flattened dictionary for W&B.
    """
    predictions_df['image'] = predictions_df['image'].astype(str)
    ground_truth_df['image'] = ground_truth_df['image'].astype(str)
    
    merged_df = pd.merge(predictions_df, ground_truth_df, on='image', how='inner')
    if merged_df.empty:
        log.warning("No matching images found between predictions and ground truth. Skipping metrics.")
        return {}

    y_true = merged_df['true_emotion']
    y_pred = merged_df['predicted_emotion']

    log.info("\n" + classification_report(y_true, y_pred, zero_division=0))

    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    metrics_to_log = {}
    metrics_to_log['num_samples'] = len(merged_df)

    for class_or_avg, scores in report.items():
        if isinstance(scores, dict):  
            for metric, value in scores.items():
                metrics_to_log[f"{class_or_avg.replace(' ', '_')}_{metric}"] = value
        else:  
            metrics_to_log[class_or_avg] = scores
            
    return metrics_to_log

class DataHandler:
    """Handles all data loading operations."""
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def load_dataset(self):
        log.info(f"Loading dataset from Hugging Face: {self.cfg.dataset.hf_path}")
        return load_dataset(self.cfg.dataset.hf_path)['train']

    def load_ground_truth(self) -> pd.DataFrame:
        gt_path = self.cfg.dataset.ground_truth_csv_path
        try:
            log.info(f"Loading ground truth from: {gt_path}")
            return pd.read_csv(gt_path)
        except FileNotFoundError:
            log.warning(f"Ground truth file not found at: {gt_path}. Evaluation will be skipped.")
            return pd.DataFrame()

    def load_emotion_definitions(self) -> dict:
        try:
            with open(self.cfg.emotion_definitions_path, 'r', encoding='utf-8') as f:
                all_definitions = json.load(f)
            return all_definitions.get(self.cfg.dataset.name, {})
        except (FileNotFoundError, json.JSONDecodeError):
            log.error("emotion_definitions.json not found or is invalid. Cannot proceed.")
            raise

class ModelEvaluator:
    """Handles model initialization, prompt creation, and prediction."""
    def __init__(self, cfg: DictConfig):
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
            log.info(f"Initializing VLLM model: {self.model_cfg.model_path}")
            self.llm = LLM(model=self.model_cfg.model_path, **self.model_cfg.vllm_args)
            self.sampling_params = SamplingParams(**self.sampling_cfg)
        
        self.prompt_templates = self._load_prompt_templates(cfg.prompt_config_path)

    def _load_prompt_templates(self, path: str) -> dict:
        log.info(f"Loading prompt templates from: {path}")
        with open(path, 'r') as f:
            return json.load(f)

    def _create_prompt(self, base64_image: str, emotion_definitions: dict) -> list:
        """Creates the full conversation prompt from templates."""
        definitions_str = "\n".join([f"- {k}: {v}" for k, v in emotion_definitions.items()])
        system_content = self.prompt_templates['system_content_template'].format(definitions_list_str=definitions_str)

        system_role = "system"
        if "gemma" in self.model_cfg.name.lower():
            system_role = "user"
        
        return [
            {"role": system_role, "content": system_content},
            {"role": "assistant", "content": self.prompt_templates['assistant_content']},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": self.prompt_templates['user_text']}
                ]
            }
        ]

    def predict(self, dataset, emotion_definitions: dict) -> pd.DataFrame:
        """Runs inference on the entire dataset."""
        results = []
        for item in tqdm(dataset, desc="Processing images"):
            base64_image = encode_image(item['image'])
            prompt = self._create_prompt(base64_image, emotion_definitions)

            if ("gpt" in self.model_cfg.name) or ("gemini" in self.model_cfg.name):
                try:
                    if "gpt" in self.model_cfg.name:
                        outputs = self.llm.chat.completions.create(
                            model=self.model_cfg.name,
                            messages=prompt,
                            max_tokens=self.sampling_cfg.max_tokens,
                            temperature=self.sampling_cfg.temperature,
                            seed=self.sampling_cfg.seed,
                        )
                    elif "gemini" in self.model_cfg.name:
                        outputs = self.llm.chat.completions.create(
                            model=self.model_cfg.name,
                            messages=prompt,
                            max_tokens=self.sampling_cfg.max_tokens,
                            temperature=self.sampling_cfg.temperature,
                            reasoning_effort="none"
                        )
                    if (outputs.choices[0].message.content is None) or \
                        (outputs.choices[0].message.content.strip().lower() == "i'm sorry, but i can't assist with that request"):
                        predicted_text = "BLOCKED"
                    else:
                        predicted_text = outputs.choices[0].message.content.strip().lower()
                except Exception as e:
                    log.error(f"Error during API call: {e}")
                    predicted_text = "BLOCKED"
            else:
                outputs = self.llm.chat(prompt, sampling_params=self.sampling_params, use_tqdm=False)
                predicted_text = outputs[0].outputs[0].text.strip().lower()

            results.append({
                'image': item['filename'],
                'predicted_emotion': predicted_text
            })
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

    def log_to_wandb(self, metrics: dict, results_df: pd.DataFrame, output_path: str, full_config: DictConfig):
        """
        Initializes a W&B run and logs all metrics, the CSV as an artifact,
        and the data as a W&B Table.
        """
        if not self.wandb_cfg:
            log.info("W&B config not found, skipping logging.")
            return

        run_name = self.wandb_cfg.run_name
        project = self.wandb_cfg.project
        log.info(f"Initializing W&B run '{run_name}' in project '{project}'")

        notes = f"Uploading evaluation results for dataset: {full_config.dataset.name} with model: {full_config.model.name}"

        with wandb.init(
            project=project,
            name=run_name,
            job_type=self.wandb_cfg.job_type,
            config=OmegaConf.to_container(full_config, resolve=True),
            notes=notes,
            reinit=False
        ) as run:
            
            # 1. Log all detailed metrics (including per-emotion)
            if metrics:
                run.log(metrics)
                log.info("Logged detailed metrics to W&B.")

            # 2. Log results CSV as a W&B Artifact for versioning
            log.info("Logging results CSV as a W&B Artifact...")
            artifact = wandb.Artifact(
                name=f'{run.name}-results',  
                type='dataset'
            )
            artifact.add_file(output_path)
            run.log_artifact(artifact)
            log.info("Artifact logging complete.")

            # 3. Log data as a W&B Table for visualization
            log.info("Logging data as a W&B Table...")
            table = wandb.Table(dataframe=results_df)
            run.log({f"{full_config.dataset.name}_results_table": table})
            log.info("Table logging complete.")

@hydra.main(version_base=None, config_path="configs", config_name="config-cls")
def main(cfg: DictConfig) -> None:
    """
    Main function to run the end-to-end emotion classification pipeline.
    """
    log.info("Starting evaluation run...")
    log.info("Full configuration for this run:\n" + OmegaConf.to_yaml(cfg))

    set_all_seeds(cfg.seed)
    # --- 1. Load Data ---
    data_handler = DataHandler(cfg)
    dataset = data_handler.load_dataset()
    
    ground_truth_df = data_handler.load_ground_truth()
    emotion_definitions = data_handler.load_emotion_definitions()

    # --- 2. Run Model Evaluation ---
    evaluator = ModelEvaluator(cfg)
    predictions_df = evaluator.predict(dataset, emotion_definitions)

    # --- 3. Calculate Metrics and Log Results ---
    metrics = {}
    if not ground_truth_df.empty:
        metrics = calculate_metrics(predictions_df, ground_truth_df)
    
    logger = ResultLogger(cfg)
    output_csv_path = logger.save_to_csv(predictions_df)
    logger.log_to_wandb(metrics, predictions_df, output_csv_path, cfg)

    log.info("Script finished successfully.")

if __name__ == "__main__":
    main()
