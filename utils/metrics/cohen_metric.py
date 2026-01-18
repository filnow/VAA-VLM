from __future__ import annotations
import argparse
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict
from scipy.stats import t
import pandas as pd


DEFAULT_CONTEXT_PATH = Path("results/LAIGAI/context/LAIGAI-demographics-context-gemma-vl-3-27b.json")
DEFAULT_REG_PATH = Path("results/LAIGAI/reg/LAIGAI-reg-gemma-vl-3-27b.csv")
DEFAULT_HUMAN_PATH = Path("data/human_context/laigai-human-context.json")
DEFAULT_OUTPUT_PATH = Path("utils/LAIGAI-context-vs-reg-cohens-d.csv")
NA_TOKENS = {"", "na", "nan", "null", "blocked"}

CONTEXT_TO_HUMAN_EMOTION: Dict[str, str] = {
    "amusement": "Amusement",
    "anger": "Anger",
    "attachment love": "Attachment_love",
    "awe": "Awe",
    "craving": "Craving",
    "disgust": "Disgust",
    "excitement": "Excitement",
    "fear": "Fear",
    "joy": "Joy",
    "neutral": "Neutral",
    "nurturant love": "Nurturant_love",
    "sadness": "Sadness",
    "positive": "Positive",
    "negative": "Negative",
    "calm": "Calm",
    "aroused": "Aroused",
    "motivated to approach": "Approach",
    "motivated to avoid": "Avoid",
}


def _to_output_label(human_emotion: str) -> str:
    return human_emotion.replace("_", " ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-emotion Cohen's d effect sizes comparing LAIGAI context "
            "results (JSON) against regression results (CSV)."
        )
    )
    parser.add_argument(
        "--context-json",
        type=Path,
        default=DEFAULT_CONTEXT_PATH,
        help=f"Path to the nested context ratings JSON (default: {DEFAULT_CONTEXT_PATH}).",
    )
    parser.add_argument(
        "--reg-csv",
        type=Path,
        default=DEFAULT_REG_PATH,
        help=f"Path to the regression summary CSV (default: {DEFAULT_REG_PATH}).",
    )
    parser.add_argument(
        "--human-json",
        type=Path,
        default=DEFAULT_HUMAN_PATH,
        help=f"Path to the human baseline annotations JSON (default: {DEFAULT_HUMAN_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV for per-emotion Cohen's d values (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=2,
        help=(
            "Minimum number of overlapping images required to compute an effect size. "
            "Emotions with fewer usable pairs are reported with NaN values."
        ),
    )
    return parser.parse_args()


def _coerce_numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return float(value)

    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        if candidate.lower() in NA_TOKENS:
            return None
        try:
            parsed = float(candidate)
        except ValueError:
            return None
        if math.isnan(parsed):
            return None
        return parsed
    return None


def load_context_predictions(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    context_predictions: Dict[str, Dict[str, Dict[str, float]]] = {}

    for image, participants in raw.items():
        if not isinstance(participants, dict):
            continue

        image_key = str(image).lower()
        per_participant: Dict[str, Dict[str, float]] = {}

        for participant, ratings in participants.items():
            if not isinstance(ratings, dict):
                continue

            participant_values: Dict[str, float] = {}
            for context_emotion, human_emotion in CONTEXT_TO_HUMAN_EMOTION.items():
                numeric = _coerce_numeric(ratings.get(context_emotion))
                if numeric is not None:
                    participant_values[context_emotion] = numeric

            if participant_values:
                per_participant[str(participant)] = participant_values

        if per_participant:
            context_predictions[image_key] = per_participant

    return context_predictions


def load_human_baseline(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    human_ratings: Dict[str, Dict[str, Dict[str, float]]] = {}

    for image, participants in raw.items():
        if not isinstance(participants, dict):
            continue

        image_key = str(image).lower()
        per_participant: Dict[str, Dict[str, float]] = {}

        for participant, ratings in participants.items():
            if not isinstance(ratings, dict):
                continue

            participant_values: Dict[str, float] = {}
            for context_emotion, human_emotion in CONTEXT_TO_HUMAN_EMOTION.items():
                numeric = _coerce_numeric(ratings.get(human_emotion))
                if numeric is not None:
                    participant_values[context_emotion] = numeric

            if participant_values:
                per_participant[str(participant)] = participant_values

        if per_participant:
            human_ratings[image_key] = per_participant

    return human_ratings


def load_regression_means(path: Path) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    df = pd.read_csv(path)
    if "image" not in df.columns:
        raise ValueError("Regression CSV must contain an 'image' column.")

    df = df.replace("BLOCKED", pd.NA)
    df["image"] = df["image"].astype(str)
    df["image_lower"] = df["image"].str.lower()

    mean_columns = [col for col in df.columns if col.endswith("_Mean")]
    for col in mean_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    regression_means: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        image_key = row["image_lower"]
        per_emotion: Dict[str, float] = {}
        for col in mean_columns:
            value = row[col]
            if pd.isna(value):
                continue
            emotion = col[:-5]  # strip trailing "_Mean"
            per_emotion[emotion] = float(value)
        if per_emotion:
            regression_means[image_key] = per_emotion

    return df, regression_means

def paired_cohens_d(differences: list[float]) -> tuple[float, float, float]:
    if not differences:
        return float("nan"), float("nan"), float("nan")

    mean_diff = stats.fmean(differences)
    if len(differences) < 2:
        return mean_diff, float("nan"), float("nan")

    std_diff = stats.stdev(differences)
    if math.isclose(std_diff, 0.0, abs_tol=1e-12):
        return mean_diff, std_diff, float("nan")

    return mean_diff, std_diff, mean_diff / std_diff


def calculate_ci(differences: list[float], confidence_level: float = 0.95) -> tuple[float, float]:
    """Calculates the confidence interval for the mean of the differences."""
    n = len(differences)
    if n < 2:
        return float("nan"), float("nan")

    mean = stats.fmean(differences)
    std_dev = stats.stdev(differences)
    
    # Calculate the standard error of the mean
    standard_error = std_dev / (n ** 0.5)
    
    # Get the critical t-value
    degrees_freedom = n - 1
    t_critical = t.ppf((1 + confidence_level) / 2, degrees_freedom)
    
    # Calculate the margin of error
    margin_of_error = t_critical * standard_error
    
    # Return the lower and upper bounds of the CI
    return mean - margin_of_error, mean + margin_of_error

def main() -> None:
    args = parse_args()

    context_predictions = load_context_predictions(args.context_json)
    human_baseline = load_human_baseline(args.human_json)
    _, regression_means = load_regression_means(args.reg_csv)
    

    errors_by_emotion = defaultdict(lambda: {"context_abs_error": [], "no_context_abs_error": []})
    
    overlapping_images = (
        set(context_predictions.keys())
        & set(human_baseline.keys())
        & set(regression_means.keys())
    )

    for image in overlapping_images:
        context_participants = context_predictions.get(image, {})
        human_participants = human_baseline.get(image, {})
        regression_emotions = regression_means.get(image, {})
        if not regression_emotions: continue
        common_participants = set(context_participants.keys()) & set(human_participants.keys())
        for participant in common_participants:
            context_ratings = context_participants.get(participant, {})
            human_ratings = human_participants.get(participant, {})
            for context_emotion, human_emotion in CONTEXT_TO_HUMAN_EMOTION.items():
                human_value = human_ratings.get(context_emotion)
                context_value = context_ratings.get(context_emotion)
                regression_value = regression_emotions.get(context_emotion)
                if human_value is None or context_value is None or regression_value is None: continue
                context_abs_error = abs(context_value - human_value)
                no_context_abs_error = abs(regression_value - human_value)
                errors_by_emotion[context_emotion]["context_abs_error"].append(context_abs_error)
                errors_by_emotion[context_emotion]["no_context_abs_error"].append(no_context_abs_error)

    records: list[dict[str, Any]] = []
    all_emotions = sorted(errors_by_emotion.keys())

    for context_emotion in all_emotions:
        human_emotion_label = CONTEXT_TO_HUMAN_EMOTION.get(context_emotion, context_emotion)
        
        context_errors = errors_by_emotion[context_emotion]["context_abs_error"]
        no_context_errors = errors_by_emotion[context_emotion]["no_context_abs_error"]
        n_pairs = len(context_errors)

        if n_pairs == 0:
            continue
            
        differences = [nc_err - c_err for nc_err, c_err in zip(no_context_errors, context_errors)]
        
        mean_diff, std_diff, cohens_d = paired_cohens_d(differences)
        
        ci_lower_diff, ci_upper_diff = calculate_ci(differences)
        
        ci_lower_d = ci_lower_diff / std_diff if std_diff > 0 else float('nan')
        ci_upper_d = ci_upper_diff / std_diff if std_diff > 0 else float('nan')

        if n_pairs < args.min_pairs:
            cohens_d = float("nan")
            ci_lower_d, ci_upper_d = float("nan"), float("nan")

        records.append({
            "emotion": _to_output_label(human_emotion_label),
            "n_pairs": n_pairs,
            "context_mae": stats.fmean(context_errors),
            "no_context_mae": stats.fmean(no_context_errors),
            "mean_error_reduction": mean_diff,
            "std_error_reduction": std_diff,
            "cohens_d": cohens_d,
            "ci_95_lower": ci_lower_d, 
            "ci_95_upper": ci_upper_d, 
        })

    results_df = pd.DataFrame(records).sort_values("emotion").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)
    
    print(f"\nSuccessfully computed Cohen's d with 95% CI for {len(results_df)} emotions.")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
