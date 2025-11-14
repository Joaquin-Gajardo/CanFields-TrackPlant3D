#!/usr/bin/env python3
"""
Batch evaluation script to recompute metrics for multiple experiments.

This script finds the latest experiment for each sequence and runs eval_metrics.py.

Usage:
    # Evaluate all latest plant sequences for a method
    python scripts/batch_eval_metrics.py --method 4DSDF --filter plants

    # Evaluate specific sequences
    python scripts/batch_eval_metrics.py --method ode_solve --sequences maize_control_plant2 sorghum_control_plant2

    # Use specific epoch
    python scripts/batch_eval_metrics.py --method NVFI --epoch 15000 --filter plants
"""

import argparse
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path


def find_latest_experiments(base_dir, sequence_filter=None, filter_type=None, experiment_type=None):
    """
    Find the latest experiment for each sequence.

    Args:
        base_dir: Base directory to search (e.g., results/pheno4d/4DSDF)
        sequence_filter: List of specific sequences to include (None = all)
        filter_type: "plants" for full plants (3 parts), "organs" for organs (4+ parts)
        experiment_type: Filter experiments by type:
            - "interp" or "interpolation": Only interpolation experiments (with _interp_stride_N)
            - "regular" or "standard": Only regular experiments (without _interp_stride_N)
            - None: All experiments (default)

    Returns:
        Dictionary mapping sequence_name -> latest_experiment_path
    """
    base_path = Path(base_dir)

    # Find all experiment directories (have timestamp prefix YYYYMMDD_HHMMSS_*)
    # Structure: base_dir/crop/sequence/YYYYMMDD_HHMMSS_*

    experiments_by_sequence = defaultdict(list)

    for crop_dir in base_path.iterdir():
        if not crop_dir.is_dir():
            continue

        for sequence_dir in crop_dir.iterdir():
            if not sequence_dir.is_dir():
                continue

            sequence_name = sequence_dir.name

            # Apply sequence filter
            if sequence_filter and sequence_name not in sequence_filter:
                continue

            # Apply type filter
            if filter_type:
                parts = sequence_name.split("_")
                num_parts = len(parts)

                if filter_type == "plants" and num_parts != 3:
                    continue
                elif filter_type == "organs" and num_parts < 4:
                    continue

            # Find experiment directories (start with timestamp)
            for exp_dir in sequence_dir.iterdir():
                if not exp_dir.is_dir():
                    continue

                exp_name = exp_dir.name

                # Check if it starts with timestamp pattern YYYYMMDD_HHMMSS
                if len(exp_name) >= 15 and exp_name[:8].isdigit() and exp_name[9:15].isdigit():
                    # Apply experiment type filter
                    is_interp = "_interp_stride_" in exp_name

                    if experiment_type:
                        if experiment_type in ["interp", "interpolation"] and not is_interp:
                            continue
                        elif experiment_type in ["regular", "standard"] and is_interp:
                            continue

                    experiments_by_sequence[sequence_name].append({
                        "path": exp_dir,
                        "timestamp": exp_name[:15],  # YYYYMMDD_HHMMSS
                        "sequence": sequence_name,
                        "crop": crop_dir.name,
                        "is_interp": is_interp
                    })

    # Get latest experiment for each sequence
    latest_experiments = {}
    for sequence_name, experiments in experiments_by_sequence.items():
        # Sort by timestamp (descending) and take the latest
        experiments.sort(key=lambda x: x["timestamp"], reverse=True)
        latest_experiments[sequence_name] = experiments[0]

    return latest_experiments


def load_data_splits(config_path="configs/data_splits.json"):
    """Load data splits config to identify test sequences."""
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


def run_eval_metrics(experiment_path, object_name, epoch, method, dataset="pheno4d", dry_run=False):
    """
    Run eval_metrics.py for a single experiment.

    Args:
        experiment_path: Path to experiment directory
        object_name: Path to object data (e.g., datasets/Pheno4D/.../sequence)
        epoch: Epoch to evaluate
        method: Method name (e.g., 4DSDF, ode_solve)
        dataset: Dataset name (default: pheno4d)
        dry_run: If True, just print commands without executing

    Returns:
        Dictionary with status and details:
        {
            "success": bool,
            "metrics_saved": bool,
            "error_type": str or None,
            "fix_command": str or None
        }
    """
    cmd = [
        "python", "eval_metrics.py",
        "-d", dataset,
        "-o", object_name,
        "-en", str(experiment_path),
        "-e", str(epoch),
        "-me", method
    ]

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running: {' '.join(cmd)}")

    if dry_run:
        return {"success": True, "metrics_saved": False, "error_type": None, "fix_command": None}

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Check if metrics file was actually created
        metrics_file = Path(experiment_path) / "metrics" / f"metrics_epoch_{epoch}.json"
        metrics_saved = metrics_file.exists()

        if not metrics_saved:
            # Check if it's a missing PLY files issue
            if "No such file or directory" in result.stderr or "manifold/epoch_" in result.stderr:
                error_type = "missing_ply"
                # Build eval_sphere command to regenerate PLY files
                fix_cmd = f"python eval_sphere.py -d {dataset} -o {object_name} -en {experiment_path} -e {epoch} -me {method} -s 2 -m False"
                print(f"⚠️  WARNING: Metrics not saved. Predicted PLY files may be missing.")
                print(f"💡 Fix: Run eval_sphere.py first to generate PLY files:")
                print(f"    {fix_cmd}")
                return {
                    "success": False,
                    "metrics_saved": False,
                    "error_type": error_type,
                    "fix_command": fix_cmd
                }
            else:
                error_type = "unknown"
                return {
                    "success": False,
                    "metrics_saved": False,
                    "error_type": error_type,
                    "fix_command": None
                }

        print(result.stdout)
        return {
            "success": True,
            "metrics_saved": True,
            "error_type": None,
            "fix_command": None
        }

    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Evaluation failed")

        # Detect error type and provide fix
        error_type = None
        fix_command = None

        stderr_text = e.stderr.lower()
        stdout_text = e.stdout.lower()

        # Check for missing PLY files
        if "no such file or directory" in stderr_text or "filenotfounderror" in stderr_text:
            if "manifold/epoch_" in stderr_text or ".ply" in stderr_text:
                error_type = "missing_ply"
                fix_command = f"python eval_sphere.py -d {dataset} -o {object_name} -en {experiment_path} -e {epoch} -me {method} -s 2 -m False"
                print(f"💡 Issue: Predicted PLY files not found")
                print(f"💡 Fix: Generate PLY files with eval_sphere.py:")
                print(f"    {fix_command}")

        # Check for missing checkpoints
        elif "checkpoint" in stderr_text or "modelparameters" in stderr_text:
            error_type = "missing_checkpoint"
            print(f"💡 Issue: Model checkpoint not found for epoch {epoch}")
            print(f"💡 Fix: Check available epochs in {experiment_path}/checkpoints/")

        # Check for missing data
        elif "ground truth" in stderr_text or object_name in stderr_text:
            error_type = "missing_data"
            print(f"💡 Issue: Ground truth data not found at {object_name}")

        else:
            error_type = "unknown"
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")

        return {
            "success": False,
            "metrics_saved": False,
            "error_type": error_type,
            "fix_command": fix_command
        }


def find_available_epochs(experiment_path):
    """
    Find all available checkpoints in an experiment directory.

    Args:
        experiment_path: Path to experiment directory

    Returns:
        List of available epoch numbers (sorted descending)
    """
    checkpoints_dir = Path(experiment_path) / "checkpoints"
    if not checkpoints_dir.exists():
        return []

    epochs = []

    # Check for checkpoints in ModelParameters subdirectory (standard structure)
    model_params_dir = checkpoints_dir / "ModelParameters"
    if model_params_dir.exists():
        for ckpt_file in model_params_dir.glob("*.pth"):
            # Extract epoch number from filename: 15000.pth
            try:
                epoch = int(ckpt_file.stem)
                epochs.append(epoch)
            except ValueError:
                continue
    else:
        # Fallback: check directly in checkpoints dir (for older structure)
        for ckpt_file in checkpoints_dir.glob("ModelParameters*.pth"):
            # Extract epoch number from filename: ModelParameters15000.pth
            try:
                epoch_str = ckpt_file.stem.replace("ModelParameters", "")
                epoch = int(epoch_str)
                epochs.append(epoch)
            except ValueError:
                continue

    # Sort descending (latest first)
    return sorted(epochs, reverse=True)


def find_best_epoch(experiment_path, requested_epoch):
    """
    Find the best available epoch to use.

    If requested_epoch is available, use it.
    Otherwise, fall back to the closest epoch before it.

    Args:
        experiment_path: Path to experiment directory
        requested_epoch: Requested epoch number

    Returns:
        Tuple of (epoch_to_use, is_fallback)
        - epoch_to_use: The epoch number to use (or None if none available)
        - is_fallback: True if we're using a fallback epoch
    """
    available_epochs = find_available_epochs(experiment_path)

    if not available_epochs:
        return None, False

    # Check if requested epoch is available
    if requested_epoch in available_epochs:
        return requested_epoch, False

    # Find closest epoch that's <= requested_epoch
    fallback_epochs = [e for e in available_epochs if e <= requested_epoch]
    if fallback_epochs:
        return fallback_epochs[0], True  # Already sorted descending

    # If no epochs before requested_epoch, return None
    return None, False


def infer_object_path(sequence_name, crop_name):
    """
    Infer the object data path from sequence and crop names.

    Args:
        sequence_name: Name of sequence (e.g., maize_control_plant2)
        crop_name: Name of crop (e.g., maize)

    Returns:
        Path to object data
    """
    # Determine if it's a plant or organ sequence
    parts = sequence_name.split("_")

    if len(parts) == 3:
        # Full plant: datasets/Pheno4D/processed/v2/plant_sequences/crop/sequence
        base = "datasets/Pheno4D/processed/v2/plant_sequences"
    else:
        # Organ: datasets/Pheno4D/processed/v2/organ_sequences/crop/sequence
        base = "datasets/Pheno4D/processed/v2/organ_sequences"

    return os.path.join(base, crop_name, sequence_name)


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluate metrics for latest experiments"
    )
    parser.add_argument(
        "--method",
        required=True,
        help="Method name (e.g., 4DSDF, NVFI, ode_solve)"
    )
    parser.add_argument(
        "--base_dir",
        default="results/pheno4d",
        help="Base results directory (default: results/pheno4d)"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=15000,
        help="Epoch to evaluate (default: 15000)"
    )
    parser.add_argument(
        "--filter",
        choices=["plants", "organs"],
        help="Filter by sequence type: 'plants' or 'organs'"
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        help="Specific sequences to evaluate (space-separated)"
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Only evaluate test split sequences from configs/data_splits.json"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--dataset",
        default="pheno4d",
        help="Dataset name (default: pheno4d)"
    )
    parser.add_argument(
        "--experiment_type",
        choices=["interp", "interpolation", "regular", "standard"],
        default=None,
        help="Filter by experiment type: 'interp'/'interpolation' for interpolation experiments, 'regular'/'standard' for standard experiments"
    )

    args = parser.parse_args()

    # Build method directory path
    method_dir = os.path.join(args.base_dir, args.method)

    if not os.path.exists(method_dir):
        print(f"Error: Method directory not found: {method_dir}")
        return

    print(f"Searching for latest experiments in: {method_dir}")
    print(f"Filter: {args.filter if args.filter else 'none'}")
    print(f"Epoch: {args.epoch}")
    print(f"Test only: {args.test_only}")
    print(f"Experiment type: {args.experiment_type if args.experiment_type else 'all'}")

    # Load data splits if test_only is requested
    test_sequences = None
    if args.test_only:
        splits = load_data_splits()
        if splits and "test" in splits:
            # The structure is: splits["test"]["plants"] and splits["test"]["sequences"]
            test_plants = set(splits["test"].get("plants", []))
            test_organ_sequences = set(splits["test"].get("sequences", []))
            # Combine both for filtering
            test_sequences = test_plants | test_organ_sequences
            print(f"Test plants: {len(test_plants)}")
            print(f"Test organ sequences: {len(test_organ_sequences)}")
            print(f"Total test sequences to evaluate: {len(test_sequences)}")
        else:
            print("Warning: Could not load test sequences, evaluating all")

    # Find latest experiments
    latest_experiments = find_latest_experiments(
        method_dir,
        sequence_filter=args.sequences,
        filter_type=args.filter,
        experiment_type=args.experiment_type
    )

    # Filter by test sequences if requested
    if test_sequences:
        latest_experiments = {
            seq: exp for seq, exp in latest_experiments.items()
            if seq in test_sequences or any(seq.startswith(ts + "_") for ts in test_sequences)
        }

    if not latest_experiments:
        print("No experiments found!")
        return

    print(f"\nFound {len(latest_experiments)} latest experiments to evaluate:")
    for seq, exp in sorted(latest_experiments.items()):
        print(f"  {seq}: {exp['path'].name}")

    # Run evaluation for each experiment
    print("\n" + "=" * 80)
    print("Starting batch evaluation...")
    print("=" * 80)

    success_count = 0
    fail_count = 0
    results = []  # Track results for each experiment

    for sequence_name, exp_info in sorted(latest_experiments.items()):
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {sequence_name}")
        print(f"Experiment: {exp_info['path'].name}")
        print(f"{'=' * 80}")

        # Infer object path
        object_path = infer_object_path(sequence_name, exp_info['crop'])

        if not args.dry_run and not os.path.exists(object_path):
            print(f"WARNING: Object path does not exist: {object_path}")
            print("Skipping...")
            fail_count += 1
            results.append({
                "sequence": sequence_name,
                "status": "failed",
                "error_type": "missing_data",
                "fix_command": None
            })
            continue

        # Find best available epoch (with fallback if requested epoch not available)
        epoch_to_use, is_fallback = find_best_epoch(exp_info['path'], args.epoch)

        if epoch_to_use is None:
            print(f"❌ ERROR: No checkpoints found in {exp_info['path']}/checkpoints/")
            print("Skipping...")
            fail_count += 1
            results.append({
                "sequence": sequence_name,
                "status": "failed",
                "error_type": "missing_checkpoint",
                "fix_command": None
            })
            continue

        if is_fallback:
            print(f"⚠️  Requested epoch {args.epoch} not available")
            print(f"📌 Using fallback epoch: {epoch_to_use}")
        else:
            print(f"✓ Using requested epoch: {epoch_to_use}")

        # Run evaluation
        result = run_eval_metrics(
            exp_info['path'],
            object_path,
            epoch_to_use,
            args.method,  # Pass method name
            dataset=args.dataset,
            dry_run=args.dry_run
        )

        # Track result
        if result["success"] and result["metrics_saved"]:
            success_count += 1
            results.append({
                "sequence": sequence_name,
                "experiment": exp_info['path'].name,
                "epoch": epoch_to_use,
                "is_fallback": is_fallback,
                "status": "success",
                "error_type": None,
                "fix_command": None
            })
        else:
            fail_count += 1
            results.append({
                "sequence": sequence_name,
                "experiment": exp_info['path'].name,
                "epoch": epoch_to_use,
                "is_fallback": is_fallback,
                "status": "failed",
                "error_type": result.get("error_type"),
                "fix_command": result.get("fix_command")
            })

    # Summary
    print("\n" + "=" * 80)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total experiments: {len(latest_experiments)}")
    print(f"✅ Successful (metrics saved): {success_count}")
    print(f"❌ Failed: {fail_count}")

    # Show fallback epochs summary
    fallback_results = [r for r in results if r.get("is_fallback") and r["status"] == "success"]
    if fallback_results:
        print(f"\n📌 Used fallback epochs (requested {args.epoch} not available): {len(fallback_results)}")
        for r in fallback_results:
            print(f"   {r['sequence']}: epoch {r['epoch']}")

    # Show failed experiments with fix commands
    failed_results = [r for r in results if r["status"] == "failed"]
    if failed_results and not args.dry_run:
        print("\n" + "=" * 80)
        print("FAILED EXPERIMENTS - FIX COMMANDS")
        print("=" * 80)

        # Group by error type
        by_error_type = {}
        for r in failed_results:
            error_type = r["error_type"] or "unknown"
            if error_type not in by_error_type:
                by_error_type[error_type] = []
            by_error_type[error_type].append(r)

        for error_type, failed_list in by_error_type.items():
            print(f"\n📋 {error_type.upper().replace('_', ' ')} ({len(failed_list)} experiments):")
            for r in failed_list:
                print(f"\n  Sequence: {r['sequence']}")
                if r["fix_command"]:
                    print(f"  Fix: {r['fix_command']}")
                else:
                    print(f"  Fix: Manual investigation required")

        # Create a shell script with all fix commands
        if any(r["fix_command"] for r in failed_results):
            fix_script_path = "fix_missing_ply_files.sh"
            with open(fix_script_path, "w") as f:
                f.write("#!/bin/bash\n")
                f.write("# Auto-generated script to fix missing PLY files\n")
                f.write("# Run this to generate missing PLY files with eval_sphere.py\n\n")
                for r in failed_results:
                    if r["fix_command"]:
                        f.write(f"# {r['sequence']}\n")
                        f.write(f"{r['fix_command']}\n\n")

            print(f"\n💾 Fix commands saved to: {fix_script_path}")
            print(f"   Run: bash {fix_script_path}")

    if args.dry_run:
        print("\n(This was a dry run - no commands were actually executed)")


if __name__ == "__main__":
    main()
