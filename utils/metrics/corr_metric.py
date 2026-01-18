import pandas as pd
from pathlib import Path

DATASETS = ["IAPS", "NAPS", "LAIGAI"]
BASE_DIR = Path(__file__).parent.parent
HUMAN_DIR = BASE_DIR / "data" / "human_reg"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = BASE_DIR / "results"

def _clean_mean_columns(df: pd.DataFrame, mean_cols: list[str]) -> pd.DataFrame:
    cleaned = df.copy()
    for c in mean_cols:
        if c in cleaned.columns:
            cleaned[c] = cleaned[c].replace("BLOCKED", pd.NA)
            cleaned[c] = cleaned[c].replace("blocked", pd.NA)
            cleaned[c] = pd.to_numeric(cleaned[c], errors="coerce")
    return cleaned


def load_human_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "image" not in df.columns:
        raise ValueError(f"'image' column not found in {path}")
    df = df.copy()
    df["image"] = df["image"].astype(str).str.lower()
    mean_cols = [c for c in df.columns if c.endswith("_Mean")]
    df = _clean_mean_columns(df, mean_cols)
    return df[["image"] + mean_cols]

def infer_model_name(csv_path: Path, dataset: str) -> str:
    name = csv_path.stem
    prefix = f"{dataset.lower()}-reg-"
    if name.lower().startswith(prefix):
        return name[len(prefix):]
    return name

def load_model_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "image" in df.columns:
        img_col = "image"
    elif "File" in df.columns: 
        img_col = "File"
        df = df.rename(columns={"File": "image"})
    else:
        raise ValueError(f"No 'image' or 'File' column in {path}")
    df = df.copy()
    df["image"] = df["image"].astype(str).str.lower()
    mean_cols = [c for c in df.columns if c.endswith("_Mean")]
    df = _clean_mean_columns(df, mean_cols)
    return df[["image"] + mean_cols]

def process_dataset(dataset: str) -> pd.DataFrame:
    """Process a single dataset and return long-format correlation DataFrame."""
    human_csv = HUMAN_DIR / f"{dataset}-human-reg.csv"
    results_dir = RESULTS_DIR / dataset / "reg"
    
    if not human_csv.exists():
        print(f"Human file not found: {human_csv}. Skipping {dataset}.")
        return pd.DataFrame()
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}. Skipping {dataset}.")
        return pd.DataFrame()
    
    human_df = load_human_df(human_csv)

    model_files = sorted(results_dir.glob(f"{dataset}-reg-*.csv"))
    if not model_files:
        print(f"No files matching {dataset}-reg-*.csv in {results_dir.resolve()}. Skipping.")
        return pd.DataFrame()

    long_rows = []

    for mf in model_files:
        model_name = infer_model_name(mf, dataset)
        try:
            model_df = load_model_df(mf)
        except Exception as e:
            print(f"Skipping {mf.name}: {e}")
            continue

        human_means = [c for c in human_df.columns if c.endswith("_Mean")]
        model_means = [c for c in model_df.columns if c.endswith("_Mean")]
        common_means = sorted(set(human_means).intersection(model_means))

        if not common_means:
            print(f"No common *_Mean columns between human and {mf.name}. Skipping.")
            continue

        merged = pd.merge(model_df, human_df, on="image", how="inner", suffixes=("_model", "_human"))
        if merged.empty:
            print(f"No matching images between human and {mf.name}. Skipping.")
            continue

        for col in common_means:
            x = merged[f"{col}_model"]
            y = merged[f"{col}_human"]
            valid = x.notna() & y.notna()
            n = int(valid.sum())
            if n < 2:
                r = float("nan")
            else:
                r = round(float(pd.Series(x[valid]).corr(pd.Series(y[valid]), method="pearson")), 2)
            long_rows.append({
                "dataset": dataset,
                "model": model_name,
                "emotion": col.replace("_Mean", ""),
                "pearson_r": r,
                "n_pairs": n,
                "result_file": mf.name,
            })

    if not long_rows:
        print(f"No correlation rows computed for {dataset}.")
        return pd.DataFrame()

    long_df = pd.DataFrame(long_rows)
    long_df = long_df.sort_values(["model", "emotion"]).reset_index(drop=True)
    return long_df


def main():
    all_long_dfs = []
    
    for dataset in DATASETS:
        print(f"\nProcessing {dataset}...")
        long_df = process_dataset(dataset)
        if not long_df.empty:
            all_long_dfs.append(long_df)
            
            # Save per-dataset outputs
            output_long = OUTPUT_DIR / dataset / f"{dataset}_reg_pearson_correlations_long.csv"
            output_wide = OUTPUT_DIR / dataset / f"{dataset}_reg_pearson_correlations_wide.csv"
            
            long_df.to_csv(output_long, index=False)
            
            # Wide pivot: one row per model, columns per emotion
            wide_df = long_df.pivot_table(index="model", columns="emotion", values="pearson_r", aggfunc="first")
            wide_df["mean_pearson_r"] = wide_df.mean(axis=1, skipna=True).round(2)
            wide_df = wide_df.sort_values("mean_pearson_r", ascending=False).reset_index()
            wide_df.to_csv(output_wide, index=False)
            
            print(f"  Saved: {output_long}")
            print(f"  Saved: {output_wide}")
    
    if not all_long_dfs:
        raise RuntimeError("No correlation rows computed for any dataset. Check inputs.")
    
    # Combine all datasets
    combined_df = pd.concat(all_long_dfs, ignore_index=True)
    
    # Average Pearson per model per dataset
    avg_pearson_df = combined_df.groupby(["dataset", "model"])["pearson_r"].mean().reset_index()
    avg_pearson_df.columns = ["dataset", "model", "avg_pearson_r"]
    avg_pearson_df["avg_pearson_r"] = avg_pearson_df["avg_pearson_r"].round(2)
    avg_pearson_df = avg_pearson_df.sort_values(["dataset", "avg_pearson_r"], ascending=[True, False]).reset_index(drop=True)
    avg_output = OUTPUT_DIR / "avg_pearson_per_model_per_dataset.csv"
    avg_pearson_df.to_csv(avg_output, index=False)
    
    print(f"\nSaved average Pearson per model per dataset to: {avg_output}")

if __name__ == "__main__":
    main()
