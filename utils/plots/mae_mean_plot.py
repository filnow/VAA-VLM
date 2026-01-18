from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd

DATASET_TITLE_MAP = {
    "LAIGAI": "LAI-GAI",
    "NAPS": "NAPS",
    "IAPS": "IAPS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate regression summary plots from existing mean tables (CSVs)."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input CSV files containing mean ratings (e.g. laigai_reg_emotion_means.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/combined_plot.png"),
        help="Path to save the generated plot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Dots-per-inch resolution for saved plots.",
    )
    parser.add_argument(
        "--hide-legend",
        action="store_true",
        help="Suppress the shared legend.",
    )
    return parser.parse_args()


def format_model_label(model: str) -> str:
    lower = model.lower()
    if lower == "human":
        return "Human Baseline"
    if lower.startswith("gpt"):
        return model.upper()
    if not model:
        return model
    return model[0].upper() + model[1:]


def format_dataset_title(dataset: str) -> str:
    return DATASET_TITLE_MAP.get(dataset.upper(), dataset.upper())


def gather_models(tables: Iterable[pd.DataFrame]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for table in tables:
        for model in table.columns.tolist():
            if model not in seen:
                seen.add(model)
                ordered.append(model)
    return ordered


def create_style_map(models: Sequence[str]) -> Dict[str, Dict[str, object]]:
    default_colors = plt.rcParams.get("axes.prop_cycle", None)
    if default_colors is not None:
        color_cycle = itertools.cycle(default_colors.by_key().get("color", []))
    else:
        color_cycle = itertools.cycle([
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ])

    marker_cycle = itertools.cycle(["o", "^", "D", "P", "X", "*", "v", "<", ">", "h"])
    style_map: Dict[str, Dict[str, object]] = {}

    for model in models:
        lower = model.lower()
        if lower == "human":
            style_map[model] = {
                "marker": "s",
                "size": 90,
                "facecolor": "none",
                "edgecolor": "#d62728",
                "linewidth": 1.5,
                "legend_facecolor": "none",
                "legend_edgecolor": "#d62728",
            }
        else:
            color = next(color_cycle)
            style_map[model] = {
                "marker": next(marker_cycle),
                "size": 60,
                "facecolor": color,
                "edgecolor": "none",
                "linewidth": 0.8,
                "color": color,
                "legend_facecolor": color,
                "legend_edgecolor": color,
            }

    return style_map


def create_legend_handles(style_map: Dict[str, Dict[str, object]]) -> Tuple[List[Line2D], List[str]]:
    handles: List[Line2D] = []
    labels: List[str] = []

    for model, style in style_map.items():
        label = format_model_label(model)
        size = max(6.0, math.sqrt(style.get("size", 70)))
        markerfacecolor = style.get("legend_facecolor", style.get("facecolor", "none"))
        markeredgecolor = style.get("legend_edgecolor", style.get("edgecolor", "none"))
        handle = Line2D(
            [0],
            [0],
            marker=style.get("marker", "o"),
            linestyle="",
            markersize=size,
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
        )
        handles.append(handle)
        labels.append(label)

    return handles, labels


def plot_dataset_panel(
    ax: plt.Axes,
    table: pd.DataFrame,
    style_map: Dict[str, Dict[str, object]],
    show_y_labels: bool,
) -> None:
    emotions = table.index.tolist()
    models = table.columns.tolist()
    base_index = pd.RangeIndex(len(emotions))
    base_positions = base_index.to_numpy()

    point_offset = 0.0

    for model in models:
        if model not in style_map:
            continue
        style = style_map[model]
        values = table[model].to_numpy()
        mask = ~pd.isna(values)
        if not mask.any():
            continue
        scatter_args = {
            "s": style.get("size", 70),
            "marker": style.get("marker", "o"),
            "facecolors": style.get("facecolor", None),
            "edgecolors": style.get("edgecolor", "none"),
            "linewidths": style.get("linewidth", 0.8),
            "color": style.get("color"),
        }

        scatter_args = {k: v for k, v in scatter_args.items() if v is not None}

        ax.scatter(
            values[mask],
            base_positions[mask] + point_offset,
            **scatter_args,
        )

    ax.set_yticks(base_positions)
    if show_y_labels:
        ax.set_yticklabels(emotions)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(left=1)
    ax.set_ylim(-0.5, len(emotions) - 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="x", linestyle="--", alpha=0.3)


def plot_combined_tables(
    dataset_tables: Sequence[Tuple[str, pd.DataFrame]],
    path: Path,
    dpi: int = 200,
    show_legend: bool = True,
) -> None:
    if not dataset_tables:
        raise ValueError("No dataset tables provided for plotting.")

    path.parent.mkdir(parents=True, exist_ok=True)

    tables = [table for _, table in dataset_tables]
    models = gather_models(tables)
    style_map = create_style_map(models)

    num_datasets = len(dataset_tables)
    max_emotions = max(table.shape[0] for table in tables)
    fig_width = 12
    panel_height = max(3.5, max_emotions * 0.35 + 1.5)
    fig_height = panel_height * num_datasets

    fig, axes = plt.subplots(
        num_datasets,
        1,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    axes_flat = axes.ravel()

    for idx, (dataset, table) in enumerate(dataset_tables):
        ax = axes_flat[idx]
        
        plot_dataset_panel(ax, table, style_map, show_y_labels=True)

    if show_legend:
        handles, labels = create_legend_handles(style_map)
        legend_cols = max(1, min(len(labels), 6))
        fig.subplots_adjust(bottom=0.23, hspace=0.4)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.56, 0.02),
            bbox_transform=fig.transFigure,
            ncol=legend_cols,
            frameon=False,
        )
    else:
        fig.subplots_adjust(bottom=0.06, hspace=0.35)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    dataset_tables: List[Tuple[str, pd.DataFrame]] = []

    for input_path in args.inputs:
        if not input_path.exists():
            print(f"Warning: File {input_path} does not exist. Skipping.")
            continue

        stem = input_path.stem
        parts = stem.split('_')
        if parts:
            dataset_name = parts[0].upper()
        else:
            dataset_name = stem.upper()

        try:
            df = pd.read_csv(input_path, index_col=0)
            dataset_tables.append((dataset_name, df))
        except Exception as e:
            print(f"Error reading {input_path}: {e}")

    if not dataset_tables:
        raise ValueError("No valid input tables found.")

    plot_combined_tables(
        dataset_tables,
        args.output,
        dpi=args.dpi,
        show_legend=not args.hide_legend
    )
    print(f"Plot saved to: {args.output}")


if __name__ == "__main__":
    main()