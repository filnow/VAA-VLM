import pandas as pd
import os
import glob
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import json
import numpy as np

DATASETS = ["IAPS", "NAPS", "LAIGAI"]
BASE_DIR = Path(__file__).parent.parent
HUMAN_DIR = BASE_DIR / "data" / "human_cls"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = BASE_DIR / "results"


def load_ground_truth_data():
    """Load ground truth data from human_cls folder."""
    ground_truth_files = {}
    
    if not HUMAN_DIR.exists():
        print(f"Warning: {HUMAN_DIR} directory not found")
        return ground_truth_files
    
    csv_files = list(HUMAN_DIR.glob("*.csv"))
    
    for csv_file in csv_files:
        filename = csv_file.name
        dataset_name = filename.replace("-human-cls.csv", "").replace("_human_cls.csv", "")
        
        try:
            df = pd.read_csv(csv_file)
            if 'image' in df.columns and 'true_emotion' in df.columns:
                ground_truth_files[dataset_name] = df
            elif 'image' in df.columns and 'emotion' in df.columns:
                df = df.rename(columns={'emotion': 'true_emotion'})
                ground_truth_files[dataset_name] = df
            else:
                print(f"Warning: {filename} doesn't have expected columns (image, true_emotion/emotion)")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return ground_truth_files

def find_classification_results():
    """Find all CSV files from results/{DATASET}/cls/ folders."""
    if not RESULTS_DIR.exists():
        print(f"Warning: {RESULTS_DIR} directory not found")
        return []
    
    cls_files = []
    for dataset in DATASETS:
        cls_dir = RESULTS_DIR / dataset / "cls"
        if cls_dir.exists():
            cls_files.extend(list(cls_dir.glob("*.csv")))
    
    return cls_files

def match_dataset_name(filename, ground_truth_keys):
    """Try to match result filename with ground truth dataset.
    
    Expected filename format: {DATASET}-cls-{MODEL}.csv
    e.g., IAPS-cls-gemini-2.5-flash.csv
    """
    filename_lower = filename.lower()
    
    if '-cls-' in filename_lower:
        dataset_part = filename.split('-cls-')[0]
        for gt_key in ground_truth_keys:
            if gt_key.lower() == dataset_part.lower():
                return gt_key
    
    for gt_key in ground_truth_keys:
        if gt_key.lower() in filename_lower:
            return gt_key
    
    for gt_key in ground_truth_keys:
        gt_parts = gt_key.lower().split('-')
        if any(part in filename_lower for part in gt_parts if len(part) > 2):
            return gt_key
    
    return None

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def should_filter_blocked(model_name):
    """Check if this model should have BLOCKED predictions filtered out."""
    model_lower = model_name.lower()
    return 'gemini' in model_lower or 'gpt' in model_lower

def calculate_comprehensive_metrics(predictions_df, ground_truth_df, model_name):
    """Calculate all classification metrics: accuracy, precision, recall, F1."""
    predictions_df['image'] = predictions_df['image'].astype(str)
    ground_truth_df['image'] = ground_truth_df['image'].astype(str)
    
    if should_filter_blocked(model_name):
        original_count = len(predictions_df)
        predictions_df = predictions_df[predictions_df['predicted_emotion'] != 'BLOCKED']
        filtered_count = original_count - len(predictions_df)
        if filtered_count > 0:
            print(f"  Filtered out {filtered_count} BLOCKED predictions for {model_name}")
    
    merged_df = pd.merge(predictions_df, ground_truth_df, on='image', how='inner')
    
    if merged_df.empty:
        print("  Warning: No matching images found between predictions and ground truth")
        return None
    
    filter_note = " (excluding BLOCKED)" if should_filter_blocked(model_name) else ""
    print(f"  Matched {len(merged_df)} images out of {len(predictions_df)} predictions{filter_note}")
    
    y_true = merged_df['true_emotion']
    y_pred = merged_df['predicted_emotion']
    emotions = sorted(y_true.unique())
    
    # Overall metrics
    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    overall_precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    overall_recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    overall_recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    overall_f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    overall_f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-emotion metrics
    per_emotion_accuracy = {}
    per_emotion_precision = {}
    per_emotion_recall = {}
    per_emotion_f1 = {}
    per_emotion_samples = {}
    
    # Calculate per-class scores
    per_class_precision = precision_score(y_true, y_pred, average=None, labels=emotions, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=emotions, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=emotions, zero_division=0)
    
    for i, emotion in enumerate(emotions):
        emotion_mask = y_true == emotion
        emotion_count = emotion_mask.sum()
        per_emotion_samples[emotion] = int(emotion_count)
        
        if emotion_count > 0:
            emotion_accuracy = accuracy_score(y_true[emotion_mask], y_pred[emotion_mask])
            per_emotion_accuracy[emotion] = round(float(emotion_accuracy), 4)
        else:
            per_emotion_accuracy[emotion] = 0.0

        per_emotion_precision[emotion] = round(float(per_class_precision[i]), 4)
        per_emotion_recall[emotion] = round(float(per_class_recall[i]), 4)
        per_emotion_f1[emotion] = round(float(per_class_f1[i]), 4)
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    return {
        'overall_accuracy': round(float(overall_accuracy), 4) * 100,
        'overall_precision_macro': round(float(overall_precision_macro), 4) * 100,
        'overall_precision_weighted': round(float(overall_precision_weighted), 4) * 100,
        'overall_recall_macro': round(float(overall_recall_macro), 4) * 100,
        'overall_recall_weighted': round(float(overall_recall_weighted), 4) * 100,
        'overall_f1_macro': round(float(overall_f1_macro), 4) * 100,
        'overall_f1_weighted': round(float(overall_f1_weighted), 4) * 100,
        'per_emotion_accuracy': per_emotion_accuracy,
        'per_emotion_precision': per_emotion_precision,
        'per_emotion_recall': per_emotion_recall,
        'per_emotion_f1': per_emotion_f1,
        'per_emotion_samples': per_emotion_samples,
        'num_samples': int(len(merged_df)),
        'classification_report': convert_to_serializable(report),
        'emotions': list(emotions),
        'blocked_filtered': should_filter_blocked(model_name)
    }

def main():
    print("=== Comprehensive Classification Metrics Calculator ===")
    print("Metrics: Accuracy, Precision, Recall, F1 Score")
    print("Note: Excluding 'BLOCKED' predictions only for Gemini and GPT models\n")
    
    # Load ground truth data
    print("Loading ground truth data...")
    ground_truth_data = load_ground_truth_data()
    
    if not ground_truth_data:
        print("No ground truth data found. Please check data/human_cls directory.")
        return
    
    print(f"Found ground truth for datasets: {list(ground_truth_data.keys())}")
    for dataset, df in ground_truth_data.items():
        print(f"  {dataset}: {len(df)} samples")
    print()
    
    print("Finding classification result files...")
    cls_files = find_classification_results()
    
    if not cls_files:
        print("No classification result files found in results/ directory.")
        return
    
    print(f"Found {len(cls_files)} classification result files\n")
    
    all_results = []
    
    for cls_file in cls_files:
        filename = cls_file.name
        print(f"Processing: {filename}")
        
        try:
            predictions_df = pd.read_csv(cls_file)
            
            if 'image' not in predictions_df.columns or 'predicted_emotion' not in predictions_df.columns:
                print(f"  Skipping: Missing required columns (image, predicted_emotion)")
                continue
            
            matched_dataset = match_dataset_name(filename, ground_truth_data.keys())
            
            if matched_dataset is None:
                print(f"  Warning: Could not match with any ground truth dataset")
                continue
            
            print(f"  Matched with dataset: {matched_dataset}")
            
            base_name = filename.replace('.csv', '')
            if '-cls-' in base_name:
                model_name = base_name.split('-cls-')[1]
            else:
                model_name = base_name.replace('_cls', '')
            
            ground_truth_df = ground_truth_data[matched_dataset]
            metrics = calculate_comprehensive_metrics(predictions_df, ground_truth_df, model_name)
            
            if metrics is None:
                continue
            
            result = {
                'file': filename,
                'dataset': matched_dataset,
                'model': model_name,
                'overall_accuracy': metrics['overall_accuracy'],
                'overall_precision_macro': metrics['overall_precision_macro'],
                'overall_precision_weighted': metrics['overall_precision_weighted'],
                'overall_recall_macro': metrics['overall_recall_macro'],
                'overall_recall_weighted': metrics['overall_recall_weighted'],
                'overall_f1_macro': metrics['overall_f1_macro'],
                'overall_f1_weighted': metrics['overall_f1_weighted'],
                'num_samples': metrics['num_samples'],
                'per_emotion_accuracy': metrics['per_emotion_accuracy'],
                'per_emotion_precision': metrics['per_emotion_precision'],
                'per_emotion_recall': metrics['per_emotion_recall'],
                'per_emotion_f1': metrics['per_emotion_f1'],
                'per_emotion_samples': metrics['per_emotion_samples'],
                'blocked_filtered': metrics['blocked_filtered']
            }
            
            all_results.append(result)
            
            filter_note = " (excluding BLOCKED)" if metrics['blocked_filtered'] else ""
            print(f"  Overall Accuracy: {metrics['overall_accuracy']:.2f}%{filter_note}")
            print(f"  Overall Precision (Macro): {metrics['overall_precision_macro']:.2f}%")
            print(f"  Overall Precision (Weighted): {metrics['overall_precision_weighted']:.2f}%")
            print(f"  Overall Recall (Macro): {metrics['overall_recall_macro']:.2f}%")
            print(f"  Overall Recall (Weighted): {metrics['overall_recall_weighted']:.2f}%")
            print(f"  Overall F1 (Macro): {metrics['overall_f1_macro']:.2f}%")
            print(f"  Overall F1 (Weighted): {metrics['overall_f1_weighted']:.2f}%")
            print(f"  Sample Size: {metrics['num_samples']}")
            print("  Per-emotion metrics:")
            for emotion in metrics['emotions']:
                acc = metrics['per_emotion_accuracy'][emotion]
                prec = metrics['per_emotion_precision'][emotion]
                rec = metrics['per_emotion_recall'][emotion]
                f1 = metrics['per_emotion_f1'][emotion]
                samples = metrics['per_emotion_samples'][emotion]
                print(f"    {emotion}: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f} (n={samples})")
            print()
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}\n")
    
    if all_results:
        print("=== SUMMARY ===")
        print("Note: BLOCKED predictions excluded only for Gemini and GPT models\n")

        summary_data = []
        for result in all_results:
            summary_data.append({
                'dataset': result['dataset'],
                'model': result['model'],
                'accuracy': result['overall_accuracy'],
                'precision_macro': result['overall_precision_macro'],
                'precision_weighted': result['overall_precision_weighted'],
                'recall_macro': result['overall_recall_macro'],
                'recall_weighted': result['overall_recall_weighted'],
                'f1_macro': result['overall_f1_macro'],
                'f1_weighted': result['overall_f1_weighted'],
                'num_samples': result['num_samples'],
                'blocked_filtered': result['blocked_filtered']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(['dataset', 'f1_macro'], ascending=[True, False])
        
        emotion_summary_data = []
        for result in all_results:
            for emotion in result['per_emotion_accuracy'].keys():
                emotion_summary_data.append({
                    'dataset': result['dataset'],
                    'model': result['model'],
                    'emotion': emotion,
                    'accuracy': result['per_emotion_accuracy'][emotion],
                    'precision': result['per_emotion_precision'][emotion],
                    'recall': result['per_emotion_recall'][emotion],
                    'f1_score': result['per_emotion_f1'][emotion],
                    'num_samples': result['per_emotion_samples'][emotion],
                    'blocked_filtered': result['blocked_filtered']
                })
        
        emotion_summary_df = pd.DataFrame(emotion_summary_data)
        
        # Save per-dataset outputs
        for dataset in DATASETS:
            dataset_summary = summary_df[summary_df['dataset'] == dataset]
            dataset_emotion = emotion_summary_df[emotion_summary_df['dataset'] == dataset]
            
            if dataset_summary.empty:
                continue
            
            dataset_output_dir = OUTPUT_DIR / dataset
            dataset_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save summary metrics per dataset
            summary_file = dataset_output_dir / f"{dataset}_cls_metrics_summary.csv"
            dataset_summary.to_csv(summary_file, index=False)
            print(f"  Saved: {summary_file}")
            
            # Save per-emotion metrics per dataset
            emotion_file = dataset_output_dir / f"{dataset}_cls_per_emotion_metrics.csv"
            dataset_emotion.to_csv(emotion_file, index=False)
            print(f"  Saved: {emotion_file}")
        
        # Save combined summary to results folder
        combined_summary_file = OUTPUT_DIR / "cls_metrics_summary_all.csv"
        summary_df.to_csv(combined_summary_file, index=False)
        print(f"\nSaved combined summary to: {combined_summary_file}")
        
        combined_emotion_file = OUTPUT_DIR / "cls_per_emotion_metrics_all.csv"
        emotion_summary_df.to_csv(combined_emotion_file, index=False)
        print(f"Saved combined per-emotion metrics to: {combined_emotion_file}")
        
        # === AVERAGE METRICS ===
        print("\n=== AVERAGE METRICS ===\n")
        
        # 1. Average accuracy per dataset
        print("Average Accuracy per Dataset:")
        avg_accuracy_per_dataset = summary_df.groupby('dataset')['accuracy'].mean()
        for dataset, avg_acc in avg_accuracy_per_dataset.items():
            print(f"  {dataset}: {avg_acc:.2f}%")
        
        # 2. Micro average accuracy per model (weighted by sample size across all datasets)
        print("\nMicro Average Accuracy per Model (weighted by sample size across datasets):")
        micro_avg_per_model = []
        for model in summary_df['model'].unique():
            model_data = summary_df[summary_df['model'] == model]
            total_correct = (model_data['accuracy'] / 100 * model_data['num_samples']).sum()
            total_samples = model_data['num_samples'].sum()
            micro_avg = (total_correct / total_samples) * 100 if total_samples > 0 else 0
            micro_avg_per_model.append({'model': model, 'micro_avg_accuracy': micro_avg, 'total_samples': total_samples})
            print(f"  {model}: {micro_avg:.2f}% (n={total_samples})")
        
        micro_avg_df = pd.DataFrame(micro_avg_per_model).sort_values('micro_avg_accuracy', ascending=False)
        
        # 3. Average F1 per emotion per dataset
        print("\nAverage F1 Score per Emotion per Dataset:")
        avg_f1_per_emotion_dataset = emotion_summary_df.groupby(['dataset', 'emotion'])['f1_score'].mean()
        avg_f1_per_emotion_dataset_dict = {}
        for dataset in emotion_summary_df['dataset'].unique():
            print(f"  {dataset}:")
            avg_f1_per_emotion_dataset_dict[dataset] = {}
            dataset_emotions = avg_f1_per_emotion_dataset[dataset].sort_values(ascending=False)
            for emotion, avg_f1 in dataset_emotions.items():
                print(f"    {emotion}: {avg_f1:.4f}")
                avg_f1_per_emotion_dataset_dict[dataset][emotion] = avg_f1
        
        # 4. Weighted F1 per dataset (weighted average of f1_weighted across models by sample size)
        print("\nWeighted F1 per Dataset (weighted by sample size across models):")
        weighted_f1_per_dataset = []
        for dataset in summary_df['dataset'].unique():
            dataset_data = summary_df[summary_df['dataset'] == dataset]
            total_weighted_f1 = (dataset_data['f1_weighted'] * dataset_data['num_samples']).sum()
            total_samples = dataset_data['num_samples'].sum()
            weighted_f1 = total_weighted_f1 / total_samples if total_samples > 0 else 0
            weighted_f1_per_dataset.append({'dataset': dataset, 'weighted_f1': weighted_f1, 'total_samples': total_samples})
            print(f"  {dataset}: {weighted_f1:.2f}%")
        
        weighted_f1_df = pd.DataFrame(weighted_f1_per_dataset).sort_values('weighted_f1', ascending=False)
        
        # 5. Weighted F1 per model per dataset
        print("\nWeighted F1 per Model per Dataset:")
        weighted_f1_model_dataset = []
        for dataset in sorted(summary_df['dataset'].unique()):
            print(f"  {dataset}:")
            dataset_data = summary_df[summary_df['dataset'] == dataset].sort_values('f1_weighted', ascending=False)
            for _, row in dataset_data.iterrows():
                weighted_f1_model_dataset.append({
                    'dataset': dataset,
                    'model': row['model'],
                    'f1_weighted': row['f1_weighted'],
                    'num_samples': row['num_samples']
                })
                print(f"    {row['model']}: {row['f1_weighted']:.2f}%")
        
        weighted_f1_model_dataset_df = pd.DataFrame(weighted_f1_model_dataset)
        
        avg_metrics_data = {
            'avg_accuracy_per_dataset': avg_accuracy_per_dataset.to_dict(),
            'micro_avg_accuracy_per_model': {row['model']: row['micro_avg_accuracy'] for _, row in micro_avg_df.iterrows()},
            'avg_f1_per_emotion_per_dataset': avg_f1_per_emotion_dataset_dict,
            'weighted_f1_per_dataset': {row['dataset']: row['weighted_f1'] for _, row in weighted_f1_df.iterrows()},
            'weighted_f1_per_model_per_dataset': weighted_f1_model_dataset
        }
        
        # Save average metrics to results folder
        avg_metrics_file = OUTPUT_DIR / "cls_average_metrics_summary.json"
        with open(avg_metrics_file, 'w') as f:
            json.dump(convert_to_serializable(avg_metrics_data), f, indent=2)
        print(f"\nAverage metrics saved to: {avg_metrics_file}")
        
        model_avg_file = OUTPUT_DIR / "cls_model_average_metrics.csv"
        micro_avg_df.to_csv(model_avg_file, index=False)
        print(f"Model average metrics saved to: {model_avg_file}")
        
        weighted_f1_model_dataset_file = OUTPUT_DIR / "cls_weighted_f1_per_model_dataset.csv"
        weighted_f1_model_dataset_df.to_csv(weighted_f1_model_dataset_file, index=False)
        print(f"Weighted F1 per model per dataset saved to: {weighted_f1_model_dataset_file}")
        
        dataset_avg_df = pd.merge(
            avg_accuracy_per_dataset.reset_index().rename(columns={'accuracy': 'avg_accuracy'}),
            weighted_f1_df,
            on='dataset'
        )
        dataset_avg_file = OUTPUT_DIR / "cls_dataset_average_metrics.csv"
        dataset_avg_df.to_csv(dataset_avg_file, index=False)
        print(f"Dataset average metrics saved to: {dataset_avg_file}")
        
        # Average accuracy per model per dataset
        avg_acc_per_model_dataset = summary_df[['dataset', 'model', 'accuracy']].copy()
        avg_acc_per_model_dataset = avg_acc_per_model_dataset.rename(columns={'accuracy': 'avg_accuracy'})
        avg_acc_per_model_dataset['avg_accuracy'] = avg_acc_per_model_dataset['avg_accuracy'].round(2)
        avg_acc_per_model_dataset = avg_acc_per_model_dataset.sort_values(['dataset', 'avg_accuracy'], ascending=[True, False])
        avg_acc_output = OUTPUT_DIR / "avg_accuracy_per_model_per_dataset.csv"
        avg_acc_per_model_dataset.to_csv(avg_acc_output, index=False)
        print(f"Average accuracy per model per dataset saved to: {avg_acc_output}")
        
        # Average F1 (weighted) per model per dataset
        avg_f1_per_model_dataset = summary_df[['dataset', 'model', 'f1_weighted']].copy()
        avg_f1_per_model_dataset = avg_f1_per_model_dataset.rename(columns={'f1_weighted': 'avg_f1_weighted'})
        avg_f1_per_model_dataset['avg_f1_weighted'] = avg_f1_per_model_dataset['avg_f1_weighted'].round(2)
        avg_f1_per_model_dataset = avg_f1_per_model_dataset.sort_values(['dataset', 'avg_f1_weighted'], ascending=[True, False])
        avg_f1_output = OUTPUT_DIR / "avg_f1_weighted_per_model_per_dataset.csv"
        avg_f1_per_model_dataset.to_csv(avg_f1_output, index=False)
        print(f"Average F1 (weighted) per model per dataset saved to: {avg_f1_output}")
        
        filtered_models = summary_df[summary_df['blocked_filtered'] == True]['model'].unique()
        if len(filtered_models) > 0:
            print(f"\nModels with BLOCKED predictions filtered: {', '.join(filtered_models)}")
        
        detailed_file = OUTPUT_DIR / "cls_detailed_comprehensive_results.json"
        serializable_results = convert_to_serializable(all_results)
        with open(detailed_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nDetailed results saved to: {detailed_file}")
    else:
        print("No results to summarize.")

if __name__ == "__main__":
    main()