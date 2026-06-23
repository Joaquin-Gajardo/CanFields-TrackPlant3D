#!/usr/bin/env python3
"""
Script to check for missing or incomplete results for a given method and crop.

Usage:
    python check_missing_results.py <method> <crop> <type> [split]

Examples:
    python check_missing_results.py OF4D maize leaf
    python check_missing_results.py OF4D maize leaf test
    python check_missing_results.py NVFI sorghum plant val
    python check_missing_results.py TPSR all leaf test
"""

import json
import re
import sys
from pathlib import Path

# Sequences that need to be rerun due to split bug fix (Nov 2025)
# These had incorrect train/test splits because of timepoint mismatch
SPLIT_BUG_AFFECTED_SEQUENCES = [
    # Maize (4)
    "maize_control_plant2_leaf4",
    "maize_control_plant2_leaf5",
    "maize_control_plant3_leaf3",
    "maize_control_plant3_leaf4",
    # Sorghum (5)
    "sorghum_control_plant2_leaf3",
    "sorghum_control_plant2_leaf4",
    "sorghum_highlight_plant2_leaf3",
    "sorghum_highlight_plant2_leaf4",
    "sorghum_highlight_plant2_leaf5",
    # Tobacco (4)
    "tobacco_control_plant1_leaf5",
    "tobacco_control_plant1_leaf6",
    "tobacco_shade_plant3_leaf5",
    "tobacco_shade_plant3_leaf6",
    # Tomato (7)
    "tomato1_heat_plant3_leaf3",
    "tomato1_shade_plant1_leaf3",
    "tomato1_shade_plant1_leaf4",
    "tomato1_shade_plant1_leaf5",
    "tomato1_shade_plant1_leaf6",
    "tomato1_shade_plant1_leaf7",
    "tomato1_shade_plant1_leaf8",
]

# Sequences with very few timepoints (may produce poor results)
# Format: {sequence_name: number_of_timepoints}
FEW_TIMEPOINT_SEQUENCES = {
    # 2 timepoints
    "maize_control_plant2_leaf5": 2,
    "maize_control_plant3_leaf4": 2,
    "sorghum_highlight_plant2_leaf5": 2,
    "tobacco_control_plant1_leaf6": 2,
    # # 3 timepoints
    # "sorghum_control_plant2_leaf4": 3,
    # "sorghum_highlight_plant2_leaf4": 3,
    # "tobacco_shade_plant3_leaf6": 3,
}

# Define sequences for each crop (matching run_crop_sequences.sh)
CROP_PLANT_SEQUENCES = {
    "maize": [
        "maize_control_plant1",
        "maize_control_plant2",
        "maize_control_plant3",
    ],
    "sorghum": [
        "sorghum_control_plant1",
        "sorghum_control_plant2",
        "sorghum_highlight_plant2",
    ],
    "tobacco": [
        "tobacco_control_plant1",
        "tobacco_control_plant2",
        "tobacco_shade_plant3",
    ],
    "tomato": [
        "tomato1_heat_plant3",
        "tomato1_shade_plant1",
        "tomato2_control_plant3",
    ],
}

CROP_LEAF_SEQUENCES = {
    "maize": [
        "maize_control_plant1_leaf1",
        "maize_control_plant1_leaf2",
        "maize_control_plant1_leaf3",
        "maize_control_plant1_leaf4",
        "maize_control_plant1_stem",
        "maize_control_plant2_leaf1",
        "maize_control_plant2_leaf2",
        "maize_control_plant2_leaf3",
        "maize_control_plant2_leaf4",
        "maize_control_plant2_leaf5",
        "maize_control_plant2_stem",
        "maize_control_plant3_leaf1",
        "maize_control_plant3_leaf2",
        "maize_control_plant3_leaf3",
        "maize_control_plant3_leaf4",
        "maize_control_plant3_stem",
    ],
    "sorghum": [
        "sorghum_control_plant1_leaf1",
        "sorghum_control_plant1_leaf2",
        "sorghum_control_plant1_leaf3",
        "sorghum_control_plant1_stem",
        "sorghum_control_plant2_leaf1",
        "sorghum_control_plant2_leaf2",
        "sorghum_control_plant2_leaf3",
        "sorghum_control_plant2_leaf4",
        "sorghum_control_plant2_stem",
        "sorghum_highlight_plant2_leaf1",
        "sorghum_highlight_plant2_leaf2",
        "sorghum_highlight_plant2_leaf3",
        "sorghum_highlight_plant2_leaf4",
        "sorghum_highlight_plant2_leaf5",
        "sorghum_highlight_plant2_stem",
    ],
    "tobacco": [
        "tobacco_control_plant1_leaf1",
        "tobacco_control_plant1_leaf2",
        "tobacco_control_plant1_leaf3",
        "tobacco_control_plant1_leaf4",
        "tobacco_control_plant1_leaf5",
        "tobacco_control_plant1_leaf6",
        "tobacco_control_plant1_stem",
        "tobacco_control_plant2_leaf1",
        "tobacco_control_plant2_leaf2",
        "tobacco_control_plant2_leaf3",
        "tobacco_control_plant2_leaf4",
        "tobacco_control_plant2_leaf5",
        "tobacco_control_plant2_leaf6",
        "tobacco_control_plant2_stem",
        "tobacco_shade_plant3_leaf1",
        "tobacco_shade_plant3_leaf2",
        "tobacco_shade_plant3_leaf3",
        "tobacco_shade_plant3_leaf4",
        "tobacco_shade_plant3_leaf5",
        "tobacco_shade_plant3_leaf6",
        "tobacco_shade_plant3_stem",
    ],
    "tomato": [
        "tomato1_heat_plant3_leaf1",
        "tomato1_heat_plant3_leaf2",
        "tomato1_heat_plant3_leaf3",
        "tomato1_heat_plant3_stem",
        "tomato1_shade_plant1_leaf1",
        "tomato1_shade_plant1_leaf2",
        "tomato1_shade_plant1_leaf3",
        "tomato1_shade_plant1_leaf4",
        "tomato1_shade_plant1_leaf5",
        "tomato1_shade_plant1_leaf6",
        "tomato1_shade_plant1_leaf7",
        "tomato1_shade_plant1_leaf8",
        "tomato1_shade_plant1_stem",
        "tomato2_control_plant3_leaf1",
        "tomato2_control_plant3_leaf2",
        "tomato2_control_plant3_leaf3",
        "tomato2_control_plant3_leaf4",
        "tomato2_control_plant3_leaf5",
        "tomato2_control_plant3_leaf6",
        "tomato2_control_plant3_leaf7",
        "tomato2_control_plant3_stem",
    ],
}


def load_splits():
    """Load train/val/test splits from JSON."""
    splits_file = Path("configs/data_splits.json")
    if not splits_file.exists():
        return None

    with open(splits_file, "r") as f:
        return json.load(f)


def is_in_split(sequence_name, target_split, splits_data):
    """Check if a sequence is in the target split."""
    if target_split == "all":
        return True

    if splits_data is None:
        return True

    # Check if sequence is in the target split
    split_sequences = splits_data["splits"][target_split].get("sequences", [])
    split_plants = splits_data["splits"][target_split].get("plants", [])

    # Check exact match or if it belongs to a plant in the split
    if sequence_name in split_sequences or sequence_name in split_plants:
        return True

    return False


def get_max_epoch(result_dir):
    """Get the maximum epoch from checkpoints in a result directory."""
    checkpoint_dir = result_dir / "checkpoints" / "LatentCodes"

    if not checkpoint_dir.exists():
        return 0

    # Find all .pth files and extract epoch numbers
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    if not checkpoints:
        return 0

    epochs = []
    for ckpt in checkpoints:
        try:
            # Extract number from filename (e.g., "10000.pth" -> 10000)
            epoch = int(ckpt.stem)
            epochs.append(epoch)
        except ValueError:
            continue

    return max(epochs) if epochs else 0


def parse_run_index(experiment_name):
    """Extract the run index from an experiment folder name.

    Repeated runs are stored as sibling folders named
    "{timestamp}_run{idx}_{obj_name}". Folders without a run tag (single-run
    experiments) default to run 1.
    """
    match = re.search(r"_run(\d+)_", experiment_name)
    return int(match.group(1)) if match else 1


def check_run_dir_status(run_dir):
    """Check completeness of a single experiment folder.

    Returns (status, max_epoch) where status is "only_wandb", "incomplete" or
    "complete".
    """
    has_wandb = (run_dir / "wandb").exists()
    has_checkpoints = (run_dir / "checkpoints").exists()
    has_metrics = (run_dir / "metrics").exists()
    has_meshes = (run_dir / "meshes").exists()

    # If only wandb exists
    if has_wandb and not has_checkpoints and not has_metrics and not has_meshes:
        return "only_wandb", 0

    max_epoch = get_max_epoch(run_dir)

    if max_epoch >= 15000:
        return "complete", max_epoch
    return "incomplete", max_epoch


def check_sequence_status(method, crop, sequence_name, n_runs=1):
    """
    Check the status of a sequence (optionally across multiple runs).

    With n_runs > 1, repeated runs are stored as sibling folders tagged
    "_run{idx}_". The sequence is only "complete" when every expected run
    (run 1 .. run n_runs) is present and complete; otherwise the missing or
    incomplete runs are reported.

    Returns:
        (status, max_epoch, run_dir, runs_detail)
        - status: "missing" | "only_wandb" | "incomplete" | "complete"
        - max_epoch: max epoch of the representative run dir (0 if N/A)
        - run_dir: representative run dir (latest), or None
        - runs_detail: list of (run_idx, status, max_epoch, run_dir) for
          n_runs > 1 (empty list for the single-run case)
    """
    results_base = Path("results/pheno4d")
    sequence_dir = results_base / method / crop / sequence_name

    # Check if sequence directory exists
    if not sequence_dir.exists():
        return "missing", 0, None, []

    # Find all experiment folders (timestamped subdirectories)
    run_dirs = [d for d in sequence_dir.iterdir() if d.is_dir()]

    if not run_dirs:
        return "missing", 0, None, []

    # Single-run (default): keep backward-compatible behaviour and check the
    # most recent folder overall.
    if n_runs <= 1:
        run_dir = max(run_dirs, key=lambda d: d.name)
        status, max_epoch = check_run_dir_status(run_dir)
        return status, max_epoch, run_dir, []

    # Multi-run: group folders by run index, keep the latest per run index.
    latest_by_run = {}
    for d in run_dirs:
        idx = parse_run_index(d.name)
        if idx not in latest_by_run or d.name > latest_by_run[idx].name:
            latest_by_run[idx] = d

    runs_detail = []
    all_complete = True
    any_present = False
    for idx in range(1, n_runs + 1):
        run_dir = latest_by_run.get(idx)
        if run_dir is None:
            runs_detail.append((idx, "missing", 0, None))
            all_complete = False
            continue
        any_present = True
        status, max_epoch = check_run_dir_status(run_dir)
        runs_detail.append((idx, status, max_epoch, run_dir))
        if status != "complete":
            all_complete = False

    representative = latest_by_run.get(n_runs) or (
        max(run_dirs, key=lambda d: d.name) if run_dirs else None
    )

    if all_complete:
        return "complete", 15000, representative, runs_detail
    if not any_present:
        return "missing", 0, None, runs_detail
    return "incomplete", 0, representative, runs_detail


def check_crop_sequences(method, crop, seq_type, split, return_data=False, n_runs=1):
    """
    Check all sequences for a given crop and type.

    Args:
        return_data: If True, return data instead of printing (for "all" crops mode)
        n_runs: Expected number of runs per sequence. With n_runs > 1, a sequence
            is only complete when runs run_1 .. run_n_runs are all complete.

    Returns:
        If return_data=True: dict with sequences info, else None
    """

    # Load splits data
    splits_data = load_splits()

    # Get the appropriate sequence list
    if seq_type == "leaf":
        sequences = CROP_LEAF_SEQUENCES.get(crop, [])
        type_name = "leaf"
    else:
        sequences = CROP_PLANT_SEQUENCES.get(crop, [])
        type_name = "plant"

    if not sequences:
        print(f"No {type_name} sequences defined for crop: {crop}")
        return

    # Filter by split
    if split != "all":
        sequences = [seq for seq in sequences if is_in_split(seq, split, splits_data)]

    if not sequences:
        print(f"No sequences found for crop {crop}, type {type_name}, split {split}")
        return

    print(f"\n{'='*80}")
    runs_note = f", expecting {n_runs} runs/sequence" if n_runs > 1 else ""
    print(
        f"Checking {method} results for {crop} {type_name} sequences (split: {split}{runs_note})"
    )
    print(f"{'='*80}\n")

    missing = []
    only_wandb = []
    incomplete = []
    complete = []

    for sequence in sequences:
        status, max_epoch, run_dir, runs_detail = check_sequence_status(
            method, crop, sequence, n_runs=n_runs
        )

        if status == "missing":
            missing.append(sequence)
        elif status == "only_wandb":
            only_wandb.append((sequence, run_dir))
        elif status == "incomplete":
            incomplete.append((sequence, max_epoch, run_dir, runs_detail))
        elif status == "complete":
            complete.append(sequence)

    # Count sequences affected by split bug before printing summary
    split_bug_count = sum(1 for seq in sequences if seq in SPLIT_BUG_AFFECTED_SEQUENCES)

    # Print summary
    total = len(sequences)
    print(f"Summary:")
    print(f"  Total sequences:     {total}")
    print(f"  Complete (15000):    {len(complete)}")
    print(f"  Incomplete (<15000): {len(incomplete)}")
    print(f"  Only wandb:          {len(only_wandb)}")
    print(f"  Missing:             {len(missing)}")
    if split_bug_count > 0:
        print(f"  Need rerun (split bug): {split_bug_count}")
    print()

    # Print sequences to rerun
    to_rerun = []

    if missing:
        print(f"\nMissing sequences ({len(missing)}):")
        print("-" * 80)
        for seq in missing:
            print(f"  - {seq}")
            to_rerun.append(seq)

    if only_wandb:
        print(f"\nOnly wandb folder ({len(only_wandb)}):")
        print("-" * 80)
        for seq, run_dir in only_wandb:
            print(f"  - {seq}")
            print(f"    Path: {run_dir}")
            to_rerun.append(seq)

    if incomplete:
        print(f"\nIncomplete training ({len(incomplete)}):")
        print("-" * 80)
        for seq, max_epoch, run_dir, runs_detail in incomplete:
            if runs_detail:
                # Multi-run: show per-run breakdown
                done = sum(1 for _, st, _, _ in runs_detail if st == "complete")
                print(f"  - {seq} ({done}/{n_runs} runs complete)")
                for idx, st, ep, rdir in runs_detail:
                    if st == "complete":
                        continue
                    if st == "missing":
                        print(f"      run {idx}: MISSING")
                    else:
                        print(f"      run {idx}: {st} (max epoch: {ep}) -> {rdir}")
            else:
                print(f"  - {seq} (max epoch: {max_epoch})")
                print(f"    Path: {run_dir}")
            to_rerun.append(seq)

    # Check for sequences affected by split bug (need to be rerun even if complete)
    split_bug_sequences = []
    for seq in sequences:
        if seq in SPLIT_BUG_AFFECTED_SEQUENCES:
            # Check if already in to_rerun (missing/incomplete)
            if seq not in to_rerun:
                # Check if it has been trained
                status, max_epoch, run_dir, _ = check_sequence_status(
                    method, crop, seq, n_runs=n_runs
                )
                if status == "complete":
                    split_bug_sequences.append((seq, run_dir))
                    to_rerun.append(seq)

    if split_bug_sequences:
        print(f"\nNeed rerun due to split bug fix ({len(split_bug_sequences)}):")
        print("-" * 80)
        print("These sequences had incorrect train/test splits and need to be rerun")
        print("even though they completed training:")
        for seq, run_dir in split_bug_sequences:
            print(f"  - {seq}")
            print(f"    Path: {run_dir}")

    # If return_data mode, return the data instead of printing commands
    if return_data:
        return {
            "crop": crop,
            "to_rerun": to_rerun,
            "complete": complete,
            "incomplete": incomplete,
            "missing": missing,
            "only_wandb": only_wandb,
            "split_bug": split_bug_sequences,
        }

    # Generate commands to rerun
    if to_rerun:
        # Append the run count so reruns reproduce the expected number of runs
        runs_flag = f" --n_runs {n_runs}" if n_runs > 1 else ""
        # Filter out sequences with very few timepoints from commands
        few_timepoint_filtered = [seq for seq in to_rerun if seq in FEW_TIMEPOINT_SEQUENCES]
        to_rerun_filtered = [seq for seq in to_rerun if seq not in FEW_TIMEPOINT_SEQUENCES]

        if few_timepoint_filtered:
            print(f"\n{'='*80}")
            print(f"WARNING: Excluding sequences with very few timepoints from commands:")
            print(f"{'='*80}")
            print("These sequences may produce poor results and are NOT included in the commands below:")
            for seq in few_timepoint_filtered:
                n_timepoints = FEW_TIMEPOINT_SEQUENCES[seq]
                print(f"  - {seq:50s} ({n_timepoints} timepoints)")
            print("\nIf you want to run these anyway, use individual commands.")

        if not to_rerun_filtered:
            print(f"\nNo sequences to rerun (all are either complete or have too few timepoints).")
            print()
            return

        print(f"\n{'='*80}")
        print(f"Commands to rerun missing/incomplete sequences:")
        print(f"{'='*80}\n")

        # Split sequences into ~5 jobs (max 4-5 sequences per job)
        sequences_per_job = max(4, (len(to_rerun_filtered) + 4) // 5)  # Aim for 5 jobs

        if len(to_rerun_filtered) <= sequences_per_job:
            # Single job if few sequences
            sequences_str = " ".join(to_rerun_filtered)
            print(
                f"sbatch -S es_tang run_euler_24GB.sh {crop} {method} {type_name} {split} {sequences_str}{runs_flag}"
            )
        else:
            # Multiple jobs
            for i in range(0, len(to_rerun_filtered), sequences_per_job):
                job_sequences = to_rerun_filtered[i : i + sequences_per_job]
                sequences_str = " ".join(job_sequences)
                job_num = i // sequences_per_job + 1
                print(f"# Job {job_num} ({len(job_sequences)} sequences)")
                print(
                    f"sbatch -S es_tang run_euler_24GB.sh {crop} {method} {type_name} {split} {sequences_str}{runs_flag}"
                )
                print()

        print(f"# Or run all in one job:")
        sequences_str = " ".join(to_rerun_filtered)
        print(
            f"# sbatch -S es_tang run_euler_24GB.sh {crop} {method} {type_name} {split} {sequences_str}{runs_flag}"
        )

        print(f"\n# Or run them individually:")
        for seq in to_rerun_filtered:
            print(
                f"# sbatch -S es_tang run_euler_24GB.sh {crop} {method} {type_name} {split} {seq}{runs_flag}"
            )
    else:
        print(f"\nAll sequences are complete!")

    print()


def check_all_crops_mixed(method, seq_type, split, n_runs=1):
    """
    Check all crops and mix sequences for balanced job distribution.
    """
    all_crops = ["maize", "sorghum", "tobacco", "tomato"]
    runs_flag = f" --n_runs {n_runs}" if n_runs > 1 else ""

    # Collect data from all crops
    crop_data = []
    for crop in all_crops:
        data = check_crop_sequences(
            method, crop, seq_type, split, return_data=True, n_runs=n_runs
        )
        if data and data["to_rerun"]:
            crop_data.append(data)

    if not crop_data:
        print("\nAll sequences are complete across all crops!")
        return

    # Flatten all sequences with their crop info
    all_sequences = []
    few_timepoint_sequences = []

    for data in crop_data:
        crop = data["crop"]
        for seq in data["to_rerun"]:
            if seq in FEW_TIMEPOINT_SEQUENCES:
                few_timepoint_sequences.append((crop, seq))
            else:
                all_sequences.append((crop, seq))

    # Print warning about excluded sequences first
    if few_timepoint_sequences:
        print(f"\n{'='*80}")
        print(f"WARNING: Excluding sequences with very few timepoints from commands:")
        print(f"{'='*80}")
        print("These sequences may produce poor results and are NOT included in the commands below:")
        for crop, seq in few_timepoint_sequences:
            n_timepoints = FEW_TIMEPOINT_SEQUENCES[seq]
            print(f"  - {seq:50s} ({n_timepoints} timepoints, crop: {crop})")
        print("\nIf you want to run these anyway, use individual commands.")

    if not all_sequences:
        print(f"\nNo sequences to rerun (all are either complete or have too few timepoints).")
        return

    # Print summary for all crops
    print(f"\n{'='*80}")
    print(f"Overall Summary for {method} {seq_type} sequences (split: {split})")
    print(f"{'='*80}\n")

    total_to_rerun = len(all_sequences)
    for data in crop_data:
        crop = data["crop"]
        print(f"{crop:10s}: {len(data['to_rerun']):2d} to rerun "
              f"({len(data['complete'])} complete, {len(data['missing'])} missing, "
              f"{len(data['incomplete'])} incomplete)")

    print(f"\nTotal sequences to rerun: {total_to_rerun}")

    # Generate mixed job commands
    print(f"\n{'='*80}")
    print(f"Commands to rerun (sequences mixed across crops for balanced workload):")
    print(f"{'='*80}\n")

    # Distribute sequences across ~5 jobs using round-robin
    num_jobs = min(5, (total_to_rerun + 3) // 4)  # Aim for ~4-5 sequences per job
    jobs = [[] for _ in range(num_jobs)]

    # Round-robin distribution
    for i, (crop, seq) in enumerate(all_sequences):
        jobs[i % num_jobs].append((crop, seq))

    # Print commands for each job
    type_name = "leaf" if seq_type == "leaf" else "plant"

    for job_num, job_sequences in enumerate(jobs, 1):
        if not job_sequences:
            continue

        print(f"# Job {job_num} ({len(job_sequences)} sequences)")

        # Group by crop within this job
        crop_groups = {}
        for crop, seq in job_sequences:
            if crop not in crop_groups:
                crop_groups[crop] = []
            crop_groups[crop].append(seq)

        # Generate one sbatch command per crop in this job
        for crop in sorted(crop_groups.keys()):
            seqs = crop_groups[crop]
            sequences_str = " ".join(seqs)
            print(f"sbatch -A es_tang run_euler_24GB.sh {crop} {method} {type_name} {split} {sequences_str}{runs_flag}")

        print()

    # Print alternative: all in one go per crop
    print(f"# Or run all sequences per crop:")
    crop_all_seqs = {}
    for crop, seq in all_sequences:
        if crop not in crop_all_seqs:
            crop_all_seqs[crop] = []
        crop_all_seqs[crop].append(seq)

    for crop in sorted(crop_all_seqs.keys()):
        seqs = crop_all_seqs[crop]
        sequences_str = " ".join(seqs)
        print(f"# sbatch -A es_tang run_euler_24GB.sh {crop} {method} {type_name} {split} {sequences_str}{runs_flag}")

    print()


def main():
    # Pull the optional "--n_runs N" flag out of argv (may appear anywhere),
    # leaving the positional arguments untouched.
    argv = sys.argv[1:]
    n_runs = 1
    positional = []
    i = 0
    while i < len(argv):
        if argv[i] == "--n_runs":
            n_runs = int(argv[i + 1])
            i += 2
        elif argv[i].startswith("--n_runs="):
            n_runs = int(argv[i].split("=", 1)[1])
            i += 1
        else:
            positional.append(argv[i])
            i += 1

    if len(positional) < 3:
        print("Usage: python check_missing_results.py <method> <crop> <type> [split] [--n_runs N]")
        print("\nExamples:")
        print("  python check_missing_results.py OF4D maize leaf")
        print("  python check_missing_results.py OF4D maize leaf test")
        print("  python check_missing_results.py NVFI sorghum plant val")
        print("  python check_missing_results.py TPSR all leaf test")
        print("  python check_missing_results.py 4DSDF all plant test --n_runs 5")
        print("\nAvailable methods: OF4D, NVFI, TPSR, 4DSDF, ode_solve")
        print("Available crops: maize, sorghum, tobacco, tomato, all")
        print("Available types: plant, leaf")
        print("Available splits: test (default), val, all")
        print("--n_runs N: expected number of runs per sequence (default 1)")
        sys.exit(1)

    method = positional[0]
    crop = positional[1]
    seq_type = positional[2] if len(positional) > 2 else "plant"
    split = positional[3] if len(positional) > 3 else "test"

    if n_runs < 1:
        print(f"Error: --n_runs must be a positive integer (got {n_runs})")
        sys.exit(1)

    # Validate inputs
    if seq_type not in ["plant", "leaf"]:
        print(f"Error: Invalid type '{seq_type}'. Must be 'plant' or 'leaf'")
        sys.exit(1)

    if split not in ["test", "val", "all"]:
        print(f"Error: Invalid split '{split}'. Must be 'test', 'val', or 'all'")
        sys.exit(1)

    # Handle "all" crops
    if crop == "all":
        check_all_crops_mixed(method, seq_type, split, n_runs=n_runs)
    else:
        # Validate crop
        sequences_dict = (
            CROP_LEAF_SEQUENCES if seq_type == "leaf" else CROP_PLANT_SEQUENCES
        )
        if crop not in sequences_dict:
            print(
                f"Error: Invalid crop '{crop}'. Must be one of: {', '.join(sequences_dict.keys())}, all"
            )
            sys.exit(1)

        check_crop_sequences(method, crop, seq_type, split, n_runs=n_runs)


if __name__ == "__main__":
    main()
