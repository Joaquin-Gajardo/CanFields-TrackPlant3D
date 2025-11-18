#!/usr/bin/env python3
"""
Verify which split (train/test) leaf tips fall into.

This script checks all sequences and reports whether their tip points
are in the train split, test split, or neither.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import plyfile


def load_point_splits(split_file):
    """Load train/test point split indices from JSON."""
    if not os.path.exists(split_file):
        return None

    with open(split_file) as f:
        splits = json.load(f)

    # The split file has per-timepoint splits
    return splits


def check_tip_splits(sequence_path, split_file=None):
    """
    Check which split the tips fall into for a sequence.

    Returns:
        Dictionary with tip split information
    """
    sparse_pc_dir = Path(sequence_path) / "sparse_pc"

    if not sparse_pc_dir.exists():
        return {"error": "sparse_pc directory not found"}

    # Load splits if provided
    splits = None
    if split_file and os.path.exists(split_file):
        splits = load_point_splits(split_file)

    ply_files = sorted(sparse_pc_dir.glob("*.ply"))

    if not ply_files:
        return {"error": "No PLY files found"}

    results = {
        "sequence": sequence_path,
        "num_timesteps": len(ply_files),
        "timesteps": []
    }

    total_tips = 0
    tips_in_train = 0
    tips_in_test = 0
    tips_in_neither = 0

    for timepoint_idx, ply_file in enumerate(ply_files):
        try:
            pcd = plyfile.PlyData.read(str(ply_file))

            # Check if tips exist
            if 'is_leaf_tip' not in pcd['vertex'].data.dtype.names:
                results["timesteps"].append({
                    "file": ply_file.name,
                    "has_tip_attr": False,
                    "num_tips": 0
                })
                continue

            is_leaf_tip = pcd['vertex']['is_leaf_tip'].astype(bool)
            tip_indices = np.where(is_leaf_tip)[0]
            num_tips = len(tip_indices)

            total_tips += num_tips

            timestep_info = {
                "file": ply_file.name,
                "has_tip_attr": True,
                "num_tips": num_tips,
                "tip_indices": tip_indices.tolist()
            }

            # Check split membership if splits provided
            if splits is not None:
                # Get train/test indices for this timepoint
                timepoint_split = splits['timepoints'][timepoint_idx]
                train_set = set(timepoint_split['train_indices'])
                test_set = set(timepoint_split['test_indices'])

                tips_train = [idx for idx in tip_indices if idx in train_set]
                tips_test = [idx for idx in tip_indices if idx in test_set]
                tips_neither = [idx for idx in tip_indices
                               if idx not in train_set and idx not in test_set]

                timestep_info["tips_in_train"] = len(tips_train)
                timestep_info["tips_in_test"] = len(tips_test)
                timestep_info["tips_in_neither"] = len(tips_neither)
                timestep_info["train_tip_indices"] = tips_train
                timestep_info["test_tip_indices"] = tips_test

                tips_in_train += len(tips_train)
                tips_in_test += len(tips_test)
                tips_in_neither += len(tips_neither)

            results["timesteps"].append(timestep_info)

        except Exception as e:
            results["timesteps"].append({
                "file": ply_file.name,
                "error": str(e)
            })

    # Summary
    results["summary"] = {
        "total_tips": total_tips,
        "tips_in_train": tips_in_train,
        "tips_in_test": tips_in_test,
        "tips_in_neither": tips_in_neither
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Verify which split leaf tips fall into"
    )
    parser.add_argument(
        "sequence_path",
        help="Path to sequence directory (e.g., datasets/Pheno4D/.../maize_control_plant2_leaf2)"
    )
    parser.add_argument(
        "--split_file",
        help="Path to point split JSON file (optional)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-timestep information"
    )

    args = parser.parse_args()

    # Check sequence
    results = check_tip_splits(args.sequence_path, args.split_file)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    # Print summary
    print("=" * 80)
    print(f"Tip Split Analysis: {Path(args.sequence_path).name}")
    print("=" * 80)
    print()

    summary = results["summary"]
    print(f"Total timesteps: {results['num_timesteps']}")
    print(f"Total tips found: {summary['total_tips']}")

    if args.split_file:
        print()
        print("Split Distribution:")
        print(f"  Tips in TRAIN split: {summary['tips_in_train']}")
        print(f"  Tips in TEST split:  {summary['tips_in_test']}")
        print(f"  Tips in NEITHER:     {summary['tips_in_neither']}")
        print()

        if summary['total_tips'] > 0:
            train_pct = summary['tips_in_train'] / summary['total_tips'] * 100
            test_pct = summary['tips_in_test'] / summary['total_tips'] * 100
            print(f"  Train: {train_pct:.1f}%")
            print(f"  Test:  {test_pct:.1f}%")
            print()

            if summary['tips_in_neither'] == summary['total_tips']:
                print("⚠️  WARNING: ALL tips are in NEITHER split!")
                print("   Tips may be outside the point cloud or splits are incorrectly defined.")
            elif summary['tips_in_test'] == 0:
                print("⚠️  WARNING: ALL tips are in the TRAIN split!")
                print("   This explains why mean_tip_epe is NULL for test split.")
            elif summary['tips_in_train'] == 0:
                print("✓ All tips are in the TEST split")
            else:
                print("✓ Tips are split between train and test")

    # Detailed output
    if args.verbose:
        print()
        print("=" * 80)
        print("Per-Timestep Details:")
        print("=" * 80)

        for ts in results["timesteps"]:
            print(f"\n{ts['file']}:")
            if "error" in ts:
                print(f"  Error: {ts['error']}")
            elif not ts.get('has_tip_attr', False):
                print(f"  No tip attribute")
            else:
                print(f"  Tips: {ts['num_tips']}")
                if args.split_file and ts['num_tips'] > 0:
                    if 'tips_in_train' in ts:
                        print(f"    Train: {ts['tips_in_train']} (indices: {ts['train_tip_indices']})")
                        print(f"    Test:  {ts['tips_in_test']} (indices: {ts['test_tip_indices']})")
                    else:
                        print(f"    Tip indices: {ts['tip_indices']}")


if __name__ == "__main__":
    main()
