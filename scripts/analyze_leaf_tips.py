#!/usr/bin/env python3
"""
Analyze leaf tip availability across Pheno4D sequences.

This script checks all sequences to determine:
- Which sequences have leaf tip annotations
- How many leaf tips per timestep
- Coverage statistics for sequence selection
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm


def check_leaf_tips_in_file(ply_file):
    """
    Check if a PLY file has leaf tip annotations and count total leaves.

    Returns:
        tuple: (has_leaf_tips, num_leaf_tips, num_total_leaves)
    """
    try:
        pcd = o3d.t.io.read_point_cloud(str(ply_file))

        # Count leaf tips
        if "is_leaf_tip" in pcd.point:
            leaf_tip_mask = pcd.point.is_leaf_tip.numpy().flatten().astype(bool)
            num_tips = np.sum(leaf_tip_mask)
            has_tips = True
        else:
            num_tips = 0
            has_tips = False

        # Count total leaves from organ_label (unique IDs > 0)
        num_total_leaves = 0
        if "organ_label" in pcd.point:
            organ_labels = pcd.point.organ_label.numpy().flatten()
            unique_organs = np.unique(organ_labels)
            # Count organs with ID > 0 (exclude background)
            num_total_leaves = int(np.sum(unique_organs > 0))

        return has_tips, int(num_tips), num_total_leaves
    except Exception as e:
        print(f"Error reading {ply_file}: {e}")
        return False, 0, 0


def analyze_sequence(sequence_path, use_sparse=True):
    """
    Analyze a single sequence for leaf tip availability.

    Args:
        sequence_path: Path to sequence directory
        use_sparse: Whether to check sparse_pc or dense_pc

    Returns:
        dict: Statistics about leaf tips in this sequence
    """
    pc_folder = "sparse_pc" if use_sparse else "dense_pc"
    pc_path = sequence_path / pc_folder

    if not pc_path.exists():
        return None

    # Get all PLY files
    ply_files = sorted(list(pc_path.glob("*.ply")))

    if len(ply_files) == 0:
        return None

    # Check each file for leaf tips
    timesteps_with_tips = 0
    total_tips = 0
    tip_counts = []
    leaf_counts = []  # Total leaves per timestep from organ_labels

    for ply_file in ply_files:
        has_tips, num_tips, num_leaves = check_leaf_tips_in_file(ply_file)

        if has_tips and num_tips > 0:
            timesteps_with_tips += 1
            total_tips += num_tips
            tip_counts.append(num_tips)
        else:
            tip_counts.append(0)

        leaf_counts.append(num_leaves)

    # Calculate statistics
    # Temporal coverage: what % of timesteps have at least 1 tip
    temporal_coverage = (
        timesteps_with_tips / len(ply_files) if len(ply_files) > 0 else 0
    )

    # Leaf tip coverage: what % of total leaves have tip annotations on average
    # Use the actual leaf count from organ_labels per timestep
    if leaf_counts and max(leaf_counts) > 0:
        # Calculate per-timestep coverage as (num_tips / num_leaves) for each timestep
        per_timestep_coverage = []
        for tips, leaves in zip(tip_counts, leaf_counts):
            if leaves > 0:
                per_timestep_coverage.append(tips / leaves)
            else:
                per_timestep_coverage.append(0.0)
        avg_leaf_tip_coverage = np.mean(per_timestep_coverage)
    else:
        avg_leaf_tip_coverage = 0.0
        per_timestep_coverage = []

    avg_tips_per_timestep = total_tips / len(ply_files) if len(ply_files) > 0 else 0

    # Format tip counts as "annotated/total" for easier comparison
    tip_counts_formatted = [
        f"{tips}/{leaves}" for tips, leaves in zip(tip_counts, leaf_counts)
    ]

    max_tips = max(tip_counts) if tip_counts else 0
    max_leaves = max(leaf_counts) if leaf_counts else 0

    return {
        "sequence_name": sequence_path.name,
        "num_timesteps": len(ply_files),
        "timesteps_with_tips": timesteps_with_tips,
        "temporal_coverage": temporal_coverage,  # % of timesteps with tips
        "leaf_tip_coverage": avg_leaf_tip_coverage,  # avg % of tips present per leaf
        "total_tips": total_tips,
        "avg_tips_per_timestep": avg_tips_per_timestep,
        "min_tips": min(tip_counts) if tip_counts else 0,
        "max_tips": max_tips,
        "max_leaves": max_leaves,  # Max number of leaves from organ_labels
        "tip_counts_per_timestep": tip_counts_formatted,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze leaf tip availability in Pheno4D sequences"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets/Pheno4D/processed/v2",
        help="Path to processed Pheno4D data",
    )
    parser.add_argument(
        "--sequence-type",
        type=str,
        choices=["plant", "organ", "both"],
        default="plant",
        help="Type of sequences to analyze",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.8,
        help="Minimum leaf tip coverage threshold for selection (0-1)",
    )
    parser.add_argument(
        "--min-temporal-coverage",
        type=float,
        default=0.8,
        help="Minimum temporal coverage threshold for selection (0-1)",
    )
    parser.add_argument(
        "--min-timesteps",
        type=int,
        default=5,
        help="Minimum number of timesteps",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--top-n-per-crop",
        type=int,
        default=3,
        help="Select top N sequences per crop (default: 3)",
    )
    parser.add_argument(
        "--sequences-list",
        type=str,
        help="Output text file for selected sequence paths (one per line, for batch processing)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Determine which sequence types to analyze
    sequence_types = []
    if args.sequence_type in ["plant", "both"]:
        sequence_types.append("plant_sequences")
    if args.sequence_type in ["organ", "both"]:
        sequence_types.append("organ_sequences")

    all_results = []
    results_by_crop = defaultdict(list)

    # Analyze sequences
    for seq_type in sequence_types:
        seq_type_dir = data_dir / seq_type

        if not seq_type_dir.exists():
            print(f"Directory not found: {seq_type_dir}")
            continue

        # Iterate through crops
        for crop_dir in sorted(seq_type_dir.iterdir()):
            if not crop_dir.is_dir():
                continue

            crop_name = crop_dir.name
            print(f"\nAnalyzing {crop_name} ({seq_type})...")

            # Iterate through sequences
            sequences = sorted(list(crop_dir.iterdir()))

            for sequence_path in tqdm(sequences, desc=f"  {crop_name}"):
                if not sequence_path.is_dir():
                    continue

                result = analyze_sequence(sequence_path, use_sparse=True)

                if result is None:
                    continue

                result["crop"] = crop_name
                result["sequence_type"] = seq_type
                all_results.append(result)
                results_by_crop[crop_name].append(result)

    # Print summary
    print("\n" + "=" * 80)
    print("LEAF TIP AVAILABILITY SUMMARY")
    print("=" * 80)

    print(f"\nTotal sequences analyzed: {len(all_results)}")

    # Overall statistics
    if all_results:
        avg_temporal_coverage = np.mean([r["temporal_coverage"] for r in all_results])
        avg_leaf_tip_coverage = np.mean([r["leaf_tip_coverage"] for r in all_results])

        print(
            f"\nTemporal coverage (% timesteps with tips): {avg_temporal_coverage:.2%}"
        )
        print(f"Leaf tip coverage (% of tips present avg): {avg_leaf_tip_coverage:.2%}")

        sequences_with_full_temporal = sum(
            1 for r in all_results if r["temporal_coverage"] == 1.0
        )
        print(
            f"\nSequences with 100% temporal coverage: {sequences_with_full_temporal}/{len(all_results)}"
        )

        sequences_with_full_leaf_tips = sum(
            1 for r in all_results if r["leaf_tip_coverage"] >= 0.95
        )
        print(
            f"Sequences with ≥95% leaf tip coverage: {sequences_with_full_leaf_tips}/{len(all_results)}"
        )

        sequences_with_partial = sum(
            1 for r in all_results if 0 < r["temporal_coverage"] < 1.0
        )
        print(
            f"Sequences with partial temporal coverage: {sequences_with_partial}/{len(all_results)}"
        )

        sequences_with_no_tips = sum(
            1 for r in all_results if r["temporal_coverage"] == 0
        )
        print(
            f"Sequences with no leaf tips: {sequences_with_no_tips}/{len(all_results)}"
        )

    # Per-crop statistics
    print("\n" + "-" * 80)
    print("PER-CROP STATISTICS")
    print("-" * 80)

    for crop_name in sorted(results_by_crop.keys()):
        crop_results = results_by_crop[crop_name]
        print(f"\n{crop_name.upper()}:")
        print(f"  Total sequences: {len(crop_results)}")

        avg_temporal = np.mean([r["temporal_coverage"] for r in crop_results])
        avg_leaf_tip = np.mean([r["leaf_tip_coverage"] for r in crop_results])
        print(f"  Average temporal coverage: {avg_temporal:.2%}")
        print(f"  Average leaf tip coverage: {avg_leaf_tip:.2%}")

        full_temporal = sum(1 for r in crop_results if r["temporal_coverage"] == 1.0)
        print(
            f"  Sequences with 100% temporal coverage: {full_temporal}/{len(crop_results)}"
        )

        # Show top sequences by leaf tip coverage (most important metric)
        sorted_seqs = sorted(
            crop_results,
            key=lambda x: (
                x["leaf_tip_coverage"],
                x["temporal_coverage"],
                x["num_timesteps"],
            ),
            reverse=True,
        )

        print(f"\n  Top sequences (sorted by leaf tip coverage):")
        for i, seq in enumerate(sorted_seqs[:5]):
            print(f"    {i+1}. {seq['sequence_name']}")
            print(
                f"       Leaf tip coverage: {seq['leaf_tip_coverage']:.2%}, "
                f"Temporal: {seq['temporal_coverage']:.2%}, "
                f"Timesteps: {seq['num_timesteps']}, "
                f"Max leaves: {seq['max_leaves']}"
            )
            print(f"       Tips per timestep: {seq['tip_counts_per_timestep']}")

    # Selection recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDED SEQUENCES FOR EXPERIMENTS")
    print("=" * 80)
    print(f"\nCriteria:")
    print(f"  - Leaf tip coverage >= {args.min_coverage:.0%}")
    print(f"  - Temporal coverage >= {args.min_temporal_coverage:.0%}")
    print(f"  - Timesteps >= {args.min_timesteps}")
    print(f"  - Selecting top {args.top_n_per_crop} sequences per crop\n")

    selected_sequences = {}

    for crop_name in sorted(results_by_crop.keys()):
        crop_results = results_by_crop[crop_name]

        # Filter by criteria
        filtered = [
            r
            for r in crop_results
            if (
                r["leaf_tip_coverage"] >= args.min_coverage
                and r["temporal_coverage"] >= args.min_temporal_coverage
                and r["num_timesteps"] >= args.min_timesteps
            )
        ]

        # Sort by leaf tip coverage (primary), temporal coverage (secondary), and timesteps (tertiary)
        sorted_filtered = sorted(
            filtered,
            key=lambda x: (
                x["leaf_tip_coverage"],
                x["temporal_coverage"],
                x["num_timesteps"],
            ),
            reverse=True,
        )

        # Select top N
        top_n = sorted_filtered[: args.top_n_per_crop]
        selected_sequences[crop_name] = top_n

        print(f"{crop_name.upper()} ({len(top_n)} sequences):")
        for seq in top_n:
            print(f"  - {seq['sequence_name']}")
            print(
                f"    Leaf tip coverage: {seq['leaf_tip_coverage']:.2%}, "
                f"Temporal: {seq['temporal_coverage']:.2%}, "
                f"Timesteps: {seq['num_timesteps']}, "
                f"Max leaves: {seq['max_leaves']}"
            )
            print(f"    Tips per timestep: {seq['tip_counts_per_timestep']}")

    # Save results if requested
    if args.output:
        output_data = {
            "criteria": {
                "min_leaf_tip_coverage": args.min_coverage,
                "min_temporal_coverage": args.min_temporal_coverage,
                "min_timesteps": args.min_timesteps,
                "top_n_per_crop": args.top_n_per_crop,
            },
            "all_sequences": all_results,
            "selected_sequences": selected_sequences,
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {args.output}")

    # Save sequences list if requested (for batch processing)
    if args.sequences_list:
        sequences_list_path = Path(args.sequences_list)
        sequences_list_path.parent.mkdir(parents=True, exist_ok=True)

        with open(sequences_list_path, "w") as f:
            for crop_name in sorted(selected_sequences.keys()):
                for seq in selected_sequences[crop_name]:
                    # Build full path from data_dir
                    seq_type = seq["sequence_type"]
                    full_path = f"{args.data_dir}/{seq_type}/{crop_name}/{seq['sequence_name']}"
                    f.write(f"{full_path}\n")

        print(f"Sequence paths saved to: {args.sequences_list}")

    # Print usage example
    print("\n" + "=" * 80)
    print("EXAMPLE USAGE")
    print("=" * 80)

    if selected_sequences:
        first_crop = next(iter(selected_sequences.keys()))
        if selected_sequences[first_crop]:
            example_seq = selected_sequences[first_crop][0]
            seq_type = (
                "plant_sequences"
                if example_seq["sequence_type"] == "plant_sequences"
                else "organ_sequences"
            )
            print(f"\nTo train on a selected sequence:")
            print(
                f'  python train.py --object_name "Pheno4D/processed/v2/{seq_type}/{first_crop}/{example_seq["sequence_name"]}"'
            )


if __name__ == "__main__":
    main()
