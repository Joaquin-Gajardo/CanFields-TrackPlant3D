#!/usr/bin/env python3
"""
Helper script to aggregate metrics across multiple experiments for paper tables.

This script collects metrics from all experiments and outputs:
- CSV files for easy import into LaTeX/Excel
- Summary statistics across multiple runs
- Comparison tables by method, crop, sequence, etc.
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_run_index(experiment_name):
    """Extract the run index from an experiment name.

    Experiment names produced for repeated runs look like
    "{timestamp}_run{idx}_{obj_name}" (e.g. "20260623_120000_run2_..."). Names
    without a run tag (single-run experiments) default to run 1.

    Args:
        experiment_name: Experiment directory name

    Returns:
        Integer run index (>= 1)
    """
    match = re.search(r"_run(\d+)_", experiment_name)
    if match:
        return int(match.group(1))
    return 1


def deduplicate_to_latest_experiments(results):
    """
    Keep only the latest experiment for each method-sequence-run combination.

    When multiple experiments exist for the same method, sequence and run index,
    selects the one with the latest timestamp (extracted from experiment name).
    Different run indices are kept separately so that repeated runs (e.g. launched
    with --n_runs) survive into the aggregated table and can be averaged over.

    Args:
        results: List of result dictionaries

    Returns:
        List of deduplicated results (latest experiment per method-sequence-run)
    """
    from collections import defaultdict

    # Group by (method, sequence, split, run)
    grouped = defaultdict(list)
    for result in results:
        method = result.get("method", "unknown")
        sequence = result.get("sequence", "unknown")
        split = result.get("split", "all")
        run = result.get("run", 1)
        key = (method, sequence, split, run)
        grouped[key].append(result)

    # For each group, keep only the latest experiment
    deduplicated = []
    for key, group in grouped.items():
        if len(group) == 1:
            deduplicated.append(group[0])
        else:
            # Sort by experiment name (which starts with timestamp YYYYMMDD_HHMMSS)
            # and take the latest
            sorted_group = sorted(group, key=lambda x: x.get("experiment", ""), reverse=True)
            latest = sorted_group[0]
            deduplicated.append(latest)

    return deduplicated


def find_best_available_epoch(metrics_dir, requested_epoch, metric_type="eval"):
    """
    Find the best available epoch file, falling back to earlier epochs if needed.

    Args:
        metrics_dir: Directory containing metrics files
        requested_epoch: Requested epoch number
        metric_type: "eval" or "training"

    Returns:
        Tuple of (metrics_file_path, actual_epoch) or (None, None) if not found
    """
    if metric_type == "eval":
        pattern = "metrics_epoch_*.json"
    else:
        pattern = "training_metrics_epoch_*.json"

    # Find all epoch files in this directory
    epoch_files = list(Path(metrics_dir).glob(pattern))
    if not epoch_files:
        return None, None

    # Extract epoch numbers from filenames
    available_epochs = []
    for f in epoch_files:
        try:
            epoch_num = int(f.stem.split('_')[-1])
            available_epochs.append((epoch_num, f))
        except ValueError:
            continue

    if not available_epochs:
        return None, None

    # Sort by epoch number descending
    available_epochs.sort(reverse=True)

    # Try to find exact match first
    for epoch_num, filepath in available_epochs:
        if epoch_num == requested_epoch:
            return filepath, epoch_num

    # Fall back to closest earlier epoch
    for epoch_num, filepath in available_epochs:
        if epoch_num < requested_epoch:
            return filepath, epoch_num

    # If no earlier epoch, use the latest available
    return available_epochs[0][1], available_epochs[0][0]


def collect_metrics_from_experiments(
    base_dir="results/pheno4d",
    epoch=None,
    metric_type="eval",  # "eval" or "training"
    experiment_type=None,
):
    """
    Collect metrics from all experiments.

    Args:
        base_dir: Base results directory
        epoch: Specific epoch to collect (None = use summary or last available)
        metric_type: "eval" (from eval_metrics.py) or "training" (from training)
        experiment_type: Filter experiments by type:
            - "interp" or "interpolation": Only interpolation experiments (with _interp_stride_N)
            - "regular" or "standard": Only regular experiments (without _interp_stride_N)
            - None: All experiments (default)

    Returns:
        List of dictionaries containing experiment results
    """
    results = []
    base_path = Path(base_dir)

    # If epoch is requested, we'll search for all experiments and find best epoch for each
    if epoch:
        # Find all experiment directories (contains metrics subdirectory)
        experiment_dirs = []
        for method_dir in base_path.iterdir():
            if not method_dir.is_dir() or method_dir.name == "wandb":
                continue
            for crop_dir in method_dir.iterdir():
                if not crop_dir.is_dir():
                    continue
                for sequence_dir in crop_dir.iterdir():
                    if not sequence_dir.is_dir():
                        continue
                    for experiment_dir in sequence_dir.iterdir():
                        if not experiment_dir.is_dir():
                            continue
                        metrics_dir = experiment_dir / "metrics"
                        if metrics_dir.exists():
                            experiment_dirs.append((experiment_dir, metrics_dir))

        metrics_files = []
        for experiment_dir, metrics_dir in experiment_dirs:
            best_file, actual_epoch = find_best_available_epoch(metrics_dir, epoch, metric_type)
            if best_file:
                metrics_files.append(best_file)
    else:
        # Original behavior for summary files
        if metric_type == "eval":
            pattern = "metrics_summary.json"
        else:
            pattern = "training_metrics_epoch_*.json"

        # Find all metrics files
        metrics_files = list(base_path.rglob(pattern))

    if not metrics_files:
        print(f"\n⚠️  No metrics files found for epoch {epoch}")
        print(f"\nPossible solutions:")
        if not epoch and metric_type == "eval":
            print(
                f"  - No summary files found. Try specifying a specific epoch: --epoch 15000"
            )
            print(f"  - Or run evaluation first: python eval_metrics.py ...")
        elif epoch:
            print(f"  - Check that evaluation has been run for epoch {epoch}")
            print(f"  - Try without --epoch flag to search for all available epochs")
        print(f"  - Verify the base directory path is correct")
        return []

    for metrics_file in metrics_files:
        if "wandb" in str(metrics_file):
            continue  # Skip wandb directories

        try:
            with open(metrics_file) as f:
                data = json.load(f)

            # Extract experiment metadata from path
            # Path structure: results/pheno4d/{method}/{crop}/{sequence}/{experiment_name}/metrics/
            path_parts = metrics_file.parts
            method_idx = (
                path_parts.index("pheno4d") + 1 if "pheno4d" in path_parts else -1
            )

            if method_idx > 0 and method_idx + 3 < len(path_parts):
                method = path_parts[method_idx]
                crop = path_parts[method_idx + 1]
                sequence = path_parts[method_idx + 2]
                experiment_name = path_parts[method_idx + 3]
            else:
                method = "unknown"
                crop = "unknown"
                sequence = "unknown"
                experiment_name = metrics_file.parent.parent.name

            # Extract timestamp from experiment name (format: YYYYMMDD_HHMMSS_...)
            name_parts = experiment_name.split('_')
            if len(name_parts) >= 2 and len(name_parts[0]) == 8 and len(name_parts[1]) == 6:
                experiment_timestamp = f"{name_parts[0]}_{name_parts[1]}"
            else:
                experiment_timestamp = None

            # Extract run index ("..._run{idx}_..."); single-run experiments default to 1
            run_idx = parse_run_index(experiment_name)

            # Apply experiment type filter
            is_interp = "_interp_stride_" in experiment_name

            if experiment_type:
                if experiment_type in ["interp", "interpolation"] and not is_interp:
                    continue
                elif experiment_type in ["regular", "standard"] and is_interp:
                    continue

            # Handle summary files differently
            if "summary" in metrics_file.name:
                result = {
                    "experiment": experiment_name,
                    "method": method,
                    "crop": crop,
                    "sequence": sequence,
                    "run": run_idx,
                    "num_epochs": data.get("num_epochs_evaluated", 0),
                    "best_epoch": data.get("best_epoch_by_chamfer", None),
                    "experiment_timestamp": experiment_timestamp,
                    "metrics_file": str(metrics_file),
                }
                results.append(result)
            else:
                # Handle both old flat structure and new split structure
                has_point_splits = data.get("has_point_splits", False)

                if has_point_splits:
                    # New structure with train/test splits - create entries for both
                    # Train split
                    if "train" in data:
                        result_train = {
                            "experiment": experiment_name,
                            "method": method,
                            "crop": crop,
                            "sequence": sequence,
                            "run": run_idx,
                            "split": "train",
                            "epoch": data.get("epoch", None),
                            "mean_cd": data["train"].get("mean_chamfer_distance", None),
                            "std_cd": data["train"].get("std_chamfer_distance", None),
                            "mean_nc": data["train"].get("mean_normal_consistency", None),
                            "std_nc": data["train"].get("std_normal_consistency", None),
                            "mean_epe": data["train"].get("mean_tip_epe", None),
                            "std_epe": data["train"].get("std_tip_epe", None),
                            "n_tips_evaluated": data["train"].get("total_tips_evaluated", None),
                            "mean_accs": data["train"].get("mean_AccS", None),
                            "mean_accr": data["train"].get("mean_AccR", None),
                            "mean_epe3d_flow": data["train"].get("mean_EPE3D_flow", None),
                            "num_timesteps": data["train"].get("num_timesteps", None),
                            "storage_mb": data.get("storage_mb", None),
                            "runtime_minutes": data.get("runtime_minutes", None),
                            "metrics_timestamp": data.get("timestamp", None),
                            "experiment_timestamp": experiment_timestamp,
                            "metrics_file": str(metrics_file),
                        }
                        results.append(result_train)

                    # Test split
                    if "test" in data:
                        result_test = {
                            "experiment": experiment_name,
                            "method": method,
                            "crop": crop,
                            "sequence": sequence,
                            "run": run_idx,
                            "split": "test",
                            "epoch": data.get("epoch", None),
                            "mean_cd": data["test"].get("mean_chamfer_distance", None),
                            "std_cd": data["test"].get("std_chamfer_distance", None),
                            "mean_nc": data["test"].get("mean_normal_consistency", None),
                            "std_nc": data["test"].get("std_normal_consistency", None),
                            "mean_epe": data["test"].get("mean_tip_epe", None),
                            "std_epe": data["test"].get("std_tip_epe", None),
                            "n_tips_evaluated": data["test"].get("total_tips_evaluated", None),
                            "mean_accs": data["test"].get("mean_AccS", None),
                            "mean_accr": data["test"].get("mean_AccR", None),
                            "mean_epe3d_flow": data["test"].get("mean_EPE3D_flow", None),
                            "num_timesteps": data["test"].get("num_timesteps", None),
                            "storage_mb": data.get("storage_mb", None),
                            "runtime_minutes": data.get("runtime_minutes", None),
                            "metrics_timestamp": data.get("timestamp", None),
                            "experiment_timestamp": experiment_timestamp,
                            "metrics_file": str(metrics_file),
                        }
                        results.append(result_test)
                else:
                    # Old flat structure (backward compatible)
                    result = {
                        "experiment": experiment_name,
                        "method": method,
                        "crop": crop,
                        "sequence": sequence,
                        "run": run_idx,
                        "split": "all",
                        "epoch": data.get("epoch", None),
                        "mean_cd": data.get("mean_chamfer_distance", None),
                        "std_cd": data.get("std_chamfer_distance", None),
                        "mean_nc": data.get("mean_normal_consistency", None),
                        "std_nc": data.get("std_normal_consistency", None),
                        "mean_epe": data.get("mean_tip_epe", None),
                        "std_epe": data.get("std_tip_epe", None),
                        "n_tips_evaluated": data.get("total_tips_evaluated", None),
                        "num_timesteps": data.get("num_timesteps", None),
                        "storage_mb": data.get("storage_mb", None),
                        "runtime_minutes": data.get("runtime_minutes", None),
                        "metrics_timestamp": data.get("timestamp", None),
                        "experiment_timestamp": experiment_timestamp,
                        "metrics_file": str(metrics_file),
                    }
                    results.append(result)

        except Exception as e:
            print(f"Error reading {metrics_file}: {e}")
            continue

    return results


def load_data_splits(config_path):
    """
    Load data splits configuration from JSON file.

    Args:
        config_path: Path to data splits config file

    Returns:
        Dictionary with 'val' and 'test' splits, or None if file doesn't exist
    """
    if not os.path.exists(config_path):
        print(f"Warning: Data splits config not found at {config_path}")
        return None

    try:
        with open(config_path) as f:
            config = json.load(f)
        return config.get("splits", {})
    except Exception as e:
        print(f"Error loading data splits config: {e}")
        return None


def filter_by_data_split(results, split_type, splits_config):
    """
    Filter results by data split (val/test/all).

    Supports both exact matches and prefix matches for organ sequences:
    - If config contains "maize_control_plant1", it will match:
      - "maize_control_plant1" (exact plant match)
      - "maize_control_plant1_leaf1" (organ sequence that starts with plant name)
      - "maize_control_plant1_leaf2", etc.

    Args:
        results: List of result dictionaries
        split_type: "val", "test", or "all"
        splits_config: Dictionary with 'val' and 'test' sequence lists

    Returns:
        Filtered list of results
    """
    if split_type == "all" or splits_config is None:
        return results

    if split_type not in splits_config:
        print(f"Warning: Split type '{split_type}' not found in config")
        return results

    split_def = splits_config[split_type]
    # data_splits.json nests entries as {"plants": [...], "sequences": [...]}.
    # Flatten so both whole-plant and organ sequence names are matchable.
    if isinstance(split_def, dict):
        split_sequences = set(split_def.get("plants", []) + split_def.get("sequences", []))
    else:
        split_sequences = set(split_def)
    filtered_results = []

    for result in results:
        sequence = result.get("sequence", "")

        # Check exact match first
        if sequence in split_sequences:
            filtered_results.append(result)
        else:
            # Check if sequence starts with any plant name (for organ sequences)
            # e.g., "maize_control_plant1_leaf1" starts with "maize_control_plant1"
            for plant_name in split_sequences:
                if sequence.startswith(plant_name + "_"):
                    filtered_results.append(result)
                    break

    return filtered_results


def filter_by_sequence_type(results, sequence_type):
    """
    Filter results by sequence type (organs vs full plants).

    Uses underscore-separated parts to determine type:
    - Full plant: 3 parts (e.g., maize_control_plant1)
    - Organ: 4+ parts (e.g., maize_control_plant1_leaf1)

    Args:
        results: List of result dictionaries
        sequence_type: "organs" for organ sequences, "plants" for full plants, None for all

    Returns:
        Filtered list of results
    """
    if sequence_type is None:
        return results

    filtered_results = []
    for result in results:
        sequence = result.get("sequence", "")
        parts = sequence.split("_")
        num_parts = len(parts)

        if sequence_type == "organs" and num_parts >= 4:
            filtered_results.append(result)
        elif sequence_type == "plants" and num_parts == 3:
            filtered_results.append(result)

    return filtered_results


def create_method_by_crop_table(results):
    """
    Create a pivot table with methods in rows and crops in columns.
    Each crop has two subcolumns: CD and EPE.

    Args:
        results: List of result dictionaries

    Returns:
        pandas DataFrame with MultiIndex columns
    """
    df = pd.DataFrame(results)

    if df.empty:
        print("No results found!")
        return df

    # Need mean_cd and mean_epe
    if "mean_cd" not in df.columns or "mean_epe" not in df.columns:
        print("Missing required columns (mean_cd, mean_epe)")
        return pd.DataFrame()

    # Group by method and crop, taking the mean of metrics
    # (in case there are multiple experiments per method-crop combination)
    grouped = (
        df.groupby(["method", "crop"])
        .agg({"mean_cd": "mean", "mean_epe": "mean"})
        .reset_index()
    )

    # Create pivot table for CD
    cd_pivot = grouped.pivot(index="method", columns="crop", values="mean_cd")

    # Create pivot table for EPE
    epe_pivot = grouped.pivot(index="method", columns="crop", values="mean_epe")

    # Combine into a MultiIndex DataFrame
    # Create a list to hold all column data
    result_data = {}
    for crop in cd_pivot.columns:
        result_data[(crop, "CD")] = cd_pivot[crop]
        result_data[(crop, "EPE")] = epe_pivot[crop]

    # Create DataFrame with MultiIndex columns
    result = pd.DataFrame(result_data)

    # Sort columns by crop name
    result = result.sort_index(axis=1, level=0)

    return result


def export_method_by_crop_to_latex(df, output_file="results_method_by_crop.tex"):
    """Export method-by-crop pivot table to a well-formatted LaTeX table."""
    if df.empty:
        return

    # Get list of crops (level 0 of columns)
    crops = df.columns.get_level_values(0).unique()
    num_crops = len(crops)

    # Start building the LaTeX table
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Quantitative Results}")
    lines.append(r"\label{tab:results}")

    # Create column specification: l for method, then rr for each crop (CD, EPE)
    col_spec = "l" + "rr" * num_crops
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row: crop names with multicolumn
    header_parts = [" "]
    cmidrule_parts = []
    col_idx = 2  # Start at column 2 (column 1 is Method)
    for crop in crops:
        # Capitalize crop name
        crop_name = crop.capitalize()
        header_parts.append(rf"\multicolumn{{2}}{{c}}{{{crop_name}}}")
        # Add cmidrule for this crop's columns
        cmidrule_parts.append(rf"\cmidrule(lr){{{col_idx}-{col_idx+1}}}")
        col_idx += 2

    lines.append(" & " + " & ".join(header_parts[1:]) + r" \\")
    lines.append(" ".join(cmidrule_parts))

    # Subheader row: CD and EPE
    subheader_parts = ["Method"]
    for _ in crops:
        subheader_parts.extend(["CD", "EPE"])
    lines.append(" & ".join(subheader_parts) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for method in df.index:
        # Format method name
        if method == "ode_solve":
            method_name = "CanFields"
        else:
            method_name = method

        row_parts = [method_name]
        for crop in crops:
            cd_val = df.loc[method, (crop, "CD")]
            epe_val = df.loc[method, (crop, "EPE")]
            # Format to 2 decimal places
            row_parts.append(f"{cd_val:.2f}")
            row_parts.append(f"{epe_val:.2f}")

        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    # Write to file
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")

    print(f"LaTeX table saved to: {output_file}")


def export_to_csv(df, output_file="results.csv"):
    """Export results to CSV format."""
    df.to_csv(output_file)
    print(f"CSV file saved to: {output_file}")


def print_summary_stats(results):
    """Print summary statistics."""
    df = pd.DataFrame(results)

    if df.empty:
        print("No results found!")
        return

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    if "mean_cd" in df.columns:
        print(f"\nTotal experiments: {len(df)}")
        print(f"\nChamfer Distance (CD):")
        print(f"  Mean: {df['mean_cd'].mean():.6f} ± {df['mean_cd'].std():.6f}")
        print(f"  Min:  {df['mean_cd'].min():.6f}")
        print(f"  Max:  {df['mean_cd'].max():.6f}")

        if "mean_epe" in df.columns and df["mean_epe"].notna().any():
            print(f"\nEnd Point Error (EPE):")
            print(f"  Mean: {df['mean_epe'].mean():.6f} ± {df['mean_epe'].std():.6f}")
            print(f"  Min:  {df['mean_epe'].min():.6f}")
            print(f"  Max:  {df['mean_epe'].max():.6f}")

    # Method by crop pivot table
    if "method" in df.columns and "crop" in df.columns:
        print("\n" + "-" * 80)
        print("RESULTS: METHOD (rows) × CROP (columns)")
        print("-" * 80)
        method_crop_table = create_method_by_crop_table(results)
        if not method_crop_table.empty:
            print(method_crop_table)

    # Best/worst experiments
    if "mean_cd" in df.columns:
        print("\n" + "-" * 80)
        print("BEST EXPERIMENTS (by Chamfer Distance)")
        print("-" * 80)
        best_df = df.nsmallest(5, "mean_cd")[
            ["experiment", "method", "crop", "sequence", "mean_cd", "std_cd"]
        ]
        print(best_df.to_string(index=False))

        print("\n" + "-" * 80)
        print("WORST EXPERIMENTS (by Chamfer Distance)")
        print("-" * 80)
        worst_df = df.nlargest(5, "mean_cd")[
            ["experiment", "method", "crop", "sequence", "mean_cd", "std_cd"]
        ]
        print(worst_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate metrics from multiple experiments"
    )
    parser.add_argument(
        "--base_dir",
        default="results/pheno4d",
        help="Base results directory",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=15000,
        help="Specific epoch to collect (default: use summary or latest)",
    )
    parser.add_argument(
        "--metric_type",
        choices=["eval", "training"],
        default="eval",
        help="Type of metrics to collect",
    )
    parser.add_argument(
        "--export_csv",
        action="store_true",
        help="Export results to CSV",
    )
    parser.add_argument(
        "--export_latex",
        action="store_true",
        help="Export results to LaTeX table",
    )
    parser.add_argument(
        "--output_prefix",
        default="results",
        help="Prefix for output files",
    )
    parser.add_argument(
        "--output_suffix",
        default="",
        help="Suffix for output files (e.g., '_epoch15000')",
    )
    parser.add_argument(
        "--filter",
        choices=["organs", "plants"],
        default=None,
        help="Filter by sequence type: 'organs' for organ sequences (4+ parts), 'plants' for full plants (3 parts)",
    )
    parser.add_argument(
        "--split",
        choices=["all", "val", "test"],
        default="all",
        help="Data split to use: 'val', 'test', or 'all' (default: all)",
    )
    parser.add_argument(
        "--splits_config",
        default="configs/data_splits.json",
        help="Path to data splits config file (default: configs/data_splits.json)",
    )
    parser.add_argument(
        "--output_dir",
        default="output/tables",
        help="Output directory for exported files (default: output/tables)",
    )
    parser.add_argument(
        "--experiment_type",
        choices=["interp", "interpolation", "regular", "standard"],
        default="regular",
        help="Filter by experiment type: 'interp'/'interpolation' for interpolation experiments, 'regular'/'standard' for standard experiments",
    )

    args = parser.parse_args()

    print(f"Collecting metrics from: {args.base_dir}")
    print(f"Epoch: {args.epoch if args.epoch else 'latest/summary'}")
    print(f"Metric type: {args.metric_type}")
    print(f"Data split: {args.split}")
    print(f"Experiment type: {args.experiment_type if args.experiment_type else 'all'}")

    # Load data splits config if not using 'all'
    splits_config = None
    if args.split != "all":
        splits_config = load_data_splits(args.splits_config)
        if splits_config is None:
            print(f"Warning: Could not load splits config, using all sequences")
            args.split = "all"

    # Collect results
    results = collect_metrics_from_experiments(
        base_dir=args.base_dir,
        epoch=args.epoch,
        metric_type=args.metric_type,
        experiment_type=args.experiment_type,
    )

    if not results:
        print("No metrics found!")
        return

    print(f"\nFound {len(results)} experiments with metrics")

    # Filter by point split (train/test) if requested
    # This is different from data split (val/test sequences)
    # Check if user wants train or test point split
    split_filter = args.split if args.split in ["train", "test"] else None
    if split_filter:
        # Filter by the 'split' column in results
        results = [r for r in results if r.get("split") == split_filter]
        print(f"Filtered to {len(results)} {split_filter} point split experiments")

    # Restrict to the sequences belonging to this data split (val/test) -- e.g. the
    # 8 test plants and their organs -- so non-test sequences don't leak into the table.
    if args.split in ["val", "test"] and splits_config is not None:
        before = len(results)
        results = filter_by_data_split(results, args.split, splits_config)
        print(f"Filtered to {len(results)} results in '{args.split}' data split (was {before})")

    # Filter by sequence type if requested
    if args.filter:
        results = filter_by_sequence_type(results, args.filter)
        filter_desc = "organ-specific" if args.filter == "organs" else "full plant"
        print(f"Filtered to {len(results)} {filter_desc} experiments")

    if not results:
        print("No results after filtering!")
        return

    # Deduplicate: keep only latest experiment per method-sequence combination
    results_before_dedup = len(results)
    results = deduplicate_to_latest_experiments(results)
    if results_before_dedup > len(results):
        print(f"Deduplicated to {len(results)} experiments (removed {results_before_dedup - len(results)} older duplicates)")

    # Print summary statistics
    print_summary_stats(results)

    # Export if requested
    df = pd.DataFrame(results)

    # Create output directory if needed
    if args.export_csv or args.export_latex:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nExporting to directory: {output_dir}")

    if args.export_csv:
        # Build filename with split and filter info
        filename_parts = [args.output_prefix]
        if args.split != "all":
            filename_parts.append(args.split)
        if args.filter:
            filename_parts.append(args.filter)
        filename_parts.append(f"epoch{args.epoch}")
        if args.output_suffix:
            filename_parts.append(args.output_suffix)
        csv_file = output_dir / f"{'_'.join(filename_parts)}.csv"
        export_to_csv(df, csv_file)

    if args.export_latex:
        # Build filename with split and filter info
        filename_parts = [args.output_prefix, "method_by_crop"]
        if args.split != "all":
            filename_parts.append(args.split)
        if args.filter:
            filename_parts.append(args.filter)
        filename_parts.append(f"epoch{args.epoch}")
        if args.output_suffix:
            filename_parts.append(args.output_suffix)
        latex_file_method_crop = output_dir / f"{'_'.join(filename_parts)}.tex"

        # Create method-by-crop pivot table
        method_crop_table = create_method_by_crop_table(results)
        if not method_crop_table.empty:
            export_method_by_crop_to_latex(method_crop_table, latex_file_method_crop)


if __name__ == "__main__":
    main()
