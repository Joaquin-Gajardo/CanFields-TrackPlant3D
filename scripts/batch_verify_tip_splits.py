#!/usr/bin/env python3
"""
Batch wrapper to run verify_tip_splits.py over all organ sequences.

This script finds all organ sequences in the dataset directory and runs
the tip split verification on each, matching them with their corresponding
split JSON files.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from verify_tip_splits import check_tip_splits


def find_organ_sequences(base_dir):
    """
    Find all organ sequence directories.

    Returns:
        List of Path objects for each organ sequence directory
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: Base directory does not exist: {base_dir}")
        return []

    organ_dirs = []

    # Search for all leaf and stem directories
    for species_dir in base_path.iterdir():
        if not species_dir.is_dir():
            continue

        for organ_dir in species_dir.iterdir():
            if not organ_dir.is_dir():
                continue

            # Check if it's a leaf or stem directory
            if '_leaf' in organ_dir.name or '_stem' in organ_dir.name:
                organ_dirs.append(organ_dir)

    return sorted(organ_dirs)


def find_split_file(organ_path, splits_dir):
    """
    Find the corresponding split JSON file for an organ sequence.

    Args:
        organ_path: Path to the organ sequence directory
        splits_dir: Path to the directory containing split JSON files

    Returns:
        Path to split file if found, None otherwise
    """
    organ_name = organ_path.name
    split_filename = f"{organ_name}_split.json"
    split_path = Path(splits_dir) / split_filename

    if split_path.exists():
        return split_path

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Batch verify tip splits for all organ sequences"
    )
    parser.add_argument(
        "--organ_sequences_dir",
        default="datasets/Pheno4D/processed/v2/organ_sequences",
        help="Base directory containing organ sequences (default: datasets/Pheno4D/processed/v2/organ_sequences)"
    )
    parser.add_argument(
        "--splits_dir",
        default="configs/point_splits",
        help="Directory containing split JSON files (default: configs/point_splits)"
    )
    parser.add_argument(
        "--output",
        help="Output JSON file to save all results (optional)"
    )
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Only show summary statistics, not individual organs"
    )
    parser.add_argument(
        "--filter_species",
        help="Filter to only process specific species (e.g., 'maize', 'tobacco')"
    )
    parser.add_argument(
        "--filter_organ",
        choices=['leaf', 'stem'],
        help="Filter to only process leaves or stems"
    )
    parser.add_argument(
        "--require_split",
        action="store_true",
        help="Only process organs that have a corresponding split file"
    )

    args = parser.parse_args()

    # Find all organ sequences
    print(f"Searching for organ sequences in: {args.organ_sequences_dir}")
    organ_dirs = find_organ_sequences(args.organ_sequences_dir)

    if not organ_dirs:
        print("No organ sequences found!")
        return 1

    print(f"Found {len(organ_dirs)} organ sequences")

    # Apply filters
    if args.filter_species:
        organ_dirs = [d for d in organ_dirs if args.filter_species in str(d)]
        print(f"After species filter: {len(organ_dirs)} sequences")

    if args.filter_organ:
        organ_dirs = [d for d in organ_dirs if f'_{args.filter_organ}' in d.name]
        print(f"After organ type filter: {len(organ_dirs)} sequences")

    # Process each organ
    all_results = []
    organs_with_splits = 0
    organs_without_splits = 0
    total_tips = 0
    total_tips_in_train = 0
    total_tips_in_test = 0
    total_tips_in_neither = 0

    for i, organ_dir in enumerate(organ_dirs, 1):
        # Find corresponding split file
        split_file = find_split_file(organ_dir, args.splits_dir)

        if args.require_split and split_file is None:
            continue

        if split_file:
            organs_with_splits += 1
        else:
            organs_without_splits += 1

        # Run verification
        results = check_tip_splits(str(organ_dir), str(split_file) if split_file else None)

        if "error" in results:
            if not args.summary_only:
                print(f"[{i}/{len(organ_dirs)}] {organ_dir.name}: ERROR - {results['error']}")
            continue

        summary = results["summary"]
        total_tips += summary["total_tips"]

        if split_file:
            total_tips_in_train += summary["tips_in_train"]
            total_tips_in_test += summary["tips_in_test"]
            total_tips_in_neither += summary["tips_in_neither"]

        # Store result
        result_entry = {
            "organ": organ_dir.name,
            "species": organ_dir.parent.name,
            "path": str(organ_dir),
            "has_split_file": split_file is not None,
            "split_file": str(split_file) if split_file else None,
            "summary": summary,
            "num_timesteps": results["num_timesteps"]
        }
        all_results.append(result_entry)

        # Print progress
        if not args.summary_only:
            status = "✓" if split_file else "○"
            print(f"[{i}/{len(organ_dirs)}] {status} {organ_dir.name}")
            print(f"    Timesteps: {results['num_timesteps']}, Tips: {summary['total_tips']}")

            if split_file:
                print(f"    Train: {summary['tips_in_train']}, "
                      f"Test: {summary['tips_in_test']}, "
                      f"Neither: {summary['tips_in_neither']}")

                # Warn about issues
                if summary['total_tips'] > 0:
                    if summary['tips_in_neither'] == summary['total_tips']:
                        print(f"    ⚠️  WARNING: ALL tips in NEITHER split!")
                    elif summary['tips_in_test'] == 0:
                        print(f"    ⚠️  WARNING: ALL tips in TRAIN split!")

    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total organs processed: {len(all_results)}")
    print(f"Organs with split files: {organs_with_splits}")
    print(f"Organs without split files: {organs_without_splits}")
    print(f"\nTotal tips across all organs: {total_tips}")

    if organs_with_splits > 0:
        print(f"\nSplit distribution:")
        print(f"  Tips in TRAIN: {total_tips_in_train}")
        print(f"  Tips in TEST: {total_tips_in_test}")
        print(f"  Tips in NEITHER: {total_tips_in_neither}")

        if total_tips > 0:
            print(f"\nPercentages:")
            print(f"  Train: {total_tips_in_train / total_tips * 100:.1f}%")
            print(f"  Test: {total_tips_in_test / total_tips * 100:.1f}%")
            print(f"  Neither: {total_tips_in_neither / total_tips * 100:.1f}%")

    # Save results to JSON if requested
    if args.output:
        output_data = {
            "summary": {
                "total_organs": len(all_results),
                "organs_with_splits": organs_with_splits,
                "organs_without_splits": organs_without_splits,
                "total_tips": total_tips,
                "total_tips_in_train": total_tips_in_train,
                "total_tips_in_test": total_tips_in_test,
                "total_tips_in_neither": total_tips_in_neither
            },
            "organs": all_results
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
