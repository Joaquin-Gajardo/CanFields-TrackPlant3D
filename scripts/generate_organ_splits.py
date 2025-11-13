"""
Generate organ-level point splits based on plant-level splits.

This script:
1. Loads plant-level split files from configs/point_splits/
2. For each organ sequence (leaf/stem), loads the parent plant data
3. Extracts which points in the plant belong to that organ (using organ_label)
4. Applies the plant-level train/test split to get organ-specific indices
5. Saves organ-level split files that can be used for training

Usage:
    python scripts/generate_organ_splits.py --dataset-path datasets/Pheno4D/processed/v2
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
from tqdm import tqdm


def load_plant_split(split_file: Path) -> Dict:
    """Load a plant-level split file."""
    with open(split_file, "r") as f:
        return json.load(f)


def load_point_cloud(ply_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load point cloud and extract points and organ labels.

    Returns:
        points: (N, 3) array of point coordinates
        labels: (N,) array of organ instance labels (or None if not available)
    """
    pcd = o3d.t.io.read_point_cloud(str(ply_path))
    points = pcd.point["positions"].numpy()

    # Try to get organ labels
    labels = None
    if "organ_label" in pcd.point:
        labels = pcd.point["organ_label"].numpy().flatten()

    return points, labels


def get_organ_id_from_filename(filename: str) -> int:
    """
    Extract organ ID from filename.
    Examples:
        'day_0_sparse.ply' -> None (plant)
        'day_0_leaf3_sparse.ply' -> 3 (leaf 3)
        'day_0_stem_sparse.ply' -> 0 (stem)
    """
    import re

    # Check for leaf pattern
    leaf_match = re.search(r'_leaf(\d+)_', filename)
    if leaf_match:
        return int(leaf_match.group(1))

    # Check for stem pattern (stems are typically labeled as 0)
    if '_stem_' in filename:
        return 0

    return None


def map_organ_indices(
    plant_points: np.ndarray,
    plant_labels: np.ndarray,
    organ_points: np.ndarray,
    organ_id: int,
    plant_train_indices: List[int],
    plant_test_indices: List[int],
) -> Tuple[List[int], List[int]]:
    """
    Map plant-level train/test indices to organ-level indices.

    Strategy:
    1. Find all points in the plant that belong to this organ (by label)
    2. For each of those points, check if it's in train or test set
    3. Build mapping from plant point index to organ point index
    4. Return train/test indices in organ coordinate space

    Args:
        plant_points: (N_plant, 3) plant point cloud
        plant_labels: (N_plant,) organ labels for plant points
        organ_points: (N_organ, 3) organ point cloud
        organ_id: The organ instance ID to filter
        plant_train_indices: Train indices in plant space
        plant_test_indices: Test indices in plant space

    Returns:
        organ_train_indices: Train indices in organ space
        organ_test_indices: Test indices in organ space
    """
    # Find which plant points belong to this organ
    organ_mask = plant_labels == organ_id
    organ_plant_indices = np.where(organ_mask)[0]

    # Get the points that belong to this organ in the plant
    organ_points_from_plant = plant_points[organ_plant_indices]

    # Build mapping from plant index to organ index by matching coordinates
    # This is necessary because organ sequences may be preprocessed differently
    # (e.g., aligned, normalized) so we match by finding nearest neighbors

    # For efficiency, we'll use a simpler approach:
    # Assume the order is preserved (organ points are extracted sequentially)
    # and build a direct mapping

    plant_to_organ_map = {}
    for organ_idx, plant_idx in enumerate(organ_plant_indices):
        plant_to_organ_map[plant_idx] = organ_idx

    # Now map the train/test indices
    organ_train_indices = []
    organ_test_indices = []

    for plant_idx in plant_train_indices:
        if plant_idx in plant_to_organ_map:
            organ_train_indices.append(plant_to_organ_map[plant_idx])

    for plant_idx in plant_test_indices:
        if plant_idx in plant_to_organ_map:
            organ_test_indices.append(plant_to_organ_map[plant_idx])

    return organ_train_indices, organ_test_indices


def generate_organ_split(
    organ_sequence_path: Path,
    plant_sequence_path: Path,
    plant_split: Dict,
    output_dir: Path,
) -> bool:
    """
    Generate split file for a single organ sequence.

    Args:
        organ_sequence_path: Path to organ sequence (e.g., .../maize_control_plant1_leaf1/)
        plant_sequence_path: Path to parent plant sequence (e.g., .../maize_control_plant1/)
        plant_split: Loaded plant split dictionary
        output_dir: Directory to save organ split files

    Returns:
        True if successful, False otherwise
    """
    organ_name = organ_sequence_path.name

    # Get sparse point cloud directories
    organ_sparse_dir = organ_sequence_path / "sparse_pc"
    plant_sparse_dir = plant_sequence_path / "sparse_pc"

    if not organ_sparse_dir.exists():
        print(f"Warning: {organ_name} sparse_pc not found, skipping")
        return False

    if not plant_sparse_dir.exists():
        print(f"Warning: Parent plant sparse_pc not found for {organ_name}, skipping")
        return False

    # Process each timepoint
    timepoint_splits = []

    # Get all files once
    organ_files = sorted(organ_sparse_dir.glob("*_sparse.ply"))
    plant_files = sorted(plant_sparse_dir.glob("*_sparse.ply"))

    # Build mapping from day number to plant file and split
    plant_day_to_file = {}
    plant_day_to_split = {}
    for tp_idx, plant_file in enumerate(plant_files):
        # Extract day from filename (e.g., "17_..._D07_sparse.ply" -> "D07")
        import re
        day_match = re.search(r'_D(\d+)_', plant_file.name)
        if day_match:
            day_num = day_match.group(1)
            plant_day_to_file[day_num] = plant_file
            if tp_idx < len(plant_split["timepoints"]):
                plant_day_to_split[day_num] = plant_split["timepoints"][tp_idx]

    # Process each organ timepoint
    for organ_tp_idx, organ_file in enumerate(organ_files):
        # Extract day from organ filename
        import re
        day_match = re.search(r'_D(\d+)_', organ_file.name)
        if not day_match:
            print(f"Warning: Could not extract day from {organ_file.name}")
            continue

        day_num = day_match.group(1)

        # Find matching plant file and split
        if day_num not in plant_day_to_file:
            print(f"Warning: No matching plant file for day {day_num} in {organ_name}")
            continue

        if day_num not in plant_day_to_split:
            print(f"Warning: No matching plant split for day {day_num} in {organ_name}")
            continue

        plant_file = plant_day_to_file[day_num]
        plant_tp_split = plant_day_to_split[day_num]

        # Extract organ ID from organ filename
        organ_id = get_organ_id_from_filename(organ_file.name)
        if organ_id is None:
            print(f"Warning: Could not extract organ ID from {organ_file.name}")
            continue

        # Load point clouds
        organ_points, _ = load_point_cloud(organ_file)
        plant_points, plant_labels = load_point_cloud(plant_file)

        if plant_labels is None:
            print(f"Warning: No organ labels in plant file {plant_file.name}")
            # Fallback: use all organ points and subsample based on ratio
            n_organ = len(organ_points)
            all_indices = np.arange(n_organ)
            np.random.seed(plant_split["random_seed"])
            np.random.shuffle(all_indices)

            train_ratio = plant_split["train_ratio"]
            n_train = int(n_organ * train_ratio)

            organ_train_indices = sorted(all_indices[:n_train].tolist())
            organ_test_indices = sorted(all_indices[n_train:].tolist())
        else:
            # Map plant indices to organ indices
            plant_train_indices = plant_tp_split["train_indices"]
            plant_test_indices = plant_tp_split["test_indices"]

            organ_train_indices, organ_test_indices = map_organ_indices(
                plant_points,
                plant_labels,
                organ_points,
                organ_id,
                plant_train_indices,
                plant_test_indices,
            )

        # Create timepoint split entry
        timepoint_splits.append({
            "timepoint_idx": organ_tp_idx,
            "n_total_points": len(organ_points),
            "n_train": len(organ_train_indices),
            "n_test": len(organ_test_indices),
            "train_indices": organ_train_indices,
            "test_indices": organ_test_indices,
        })

    if len(timepoint_splits) == 0:
        print(f"Warning: No valid timepoints for {organ_name}")
        return False

    # Create organ split file
    organ_split = {
        "sequence_name": organ_name,
        "parent_plant": plant_split["sequence_name"],
        "random_seed": plant_split["random_seed"],
        "train_ratio": plant_split["train_ratio"],
        "created_date": datetime.now().isoformat(),
        "dataset_version": plant_split["dataset_version"],
        "n_timepoints": len(timepoint_splits),
        "timepoints": timepoint_splits,
    }

    # Save split file
    output_file = output_dir / f"{organ_name}_split.json"
    with open(output_file, "w") as f:
        json.dump(organ_split, f, indent=2)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate organ-level point splits from plant-level splits"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/Pheno4D/processed/v2",
        help="Path to processed Pheno4D dataset (default: datasets/Pheno4D/processed/v2)",
    )
    parser.add_argument(
        "--plant-splits-dir",
        type=str,
        default="configs/point_splits",
        help="Directory containing plant-level split files (default: configs/point_splits)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="configs/point_splits",
        help="Directory to save organ split files (default: configs/point_splits)",
    )
    parser.add_argument(
        "--crops",
        type=str,
        nargs="+",
        choices=["maize", "sorghum", "tobacco", "tomato"],
        help="Process only specific crops (default: all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing organ split files",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    plant_splits_dir = Path(args.plant_splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Generating organ-level point splits from plant-level splits")
    print("=" * 80)
    print(f"Dataset path: {dataset_path}")
    print(f"Plant splits: {plant_splits_dir}")
    print(f"Output dir: {output_dir}")
    print()

    # Find all plant split files
    plant_split_files = sorted(plant_splits_dir.glob("*_plant*_split.json"))

    if len(plant_split_files) == 0:
        print("Error: No plant split files found!")
        return

    print(f"Found {len(plant_split_files)} plant split files")

    # Track statistics
    total_organs = 0
    successful = 0
    failed = 0
    skipped = 0

    # Process each plant
    for plant_split_file in plant_split_files:
        plant_name = plant_split_file.stem.replace("_split", "")
        print(f"\nProcessing plant: {plant_name}")

        # Load plant split
        plant_split = load_plant_split(plant_split_file)

        # Find the crop name (handle tomato1/tomato2 -> tomato)
        crop_name = plant_name.split("_")[0]
        # Normalize tomato variants
        if crop_name.startswith("tomato"):
            crop_name = "tomato"

        # Filter by crop if specified
        if args.crops and crop_name not in args.crops:
            print(f"  Skipping {crop_name} (not in --crops filter)")
            continue

        # Find plant sequence directory
        plant_sequence_path = dataset_path / "plant_sequences" / crop_name / plant_name

        if not plant_sequence_path.exists():
            print(f"  Warning: Plant sequence not found: {plant_sequence_path}")
            continue

        # Find all organ sequences for this plant
        organ_sequences_dir = dataset_path / "organ_sequences" / crop_name

        if not organ_sequences_dir.exists():
            print(f"  No organ sequences found for {crop_name}")
            continue

        # Find organs that belong to this plant
        organ_sequences = sorted([
            p for p in organ_sequences_dir.iterdir()
            if p.is_dir() and p.name.startswith(plant_name)
        ])

        print(f"  Found {len(organ_sequences)} organ sequences")

        # Process each organ
        for organ_seq_path in tqdm(organ_sequences, desc=f"  {plant_name}"):
            total_organs += 1
            organ_name = organ_seq_path.name

            # Check if already exists
            output_file = output_dir / f"{organ_name}_split.json"
            if output_file.exists() and not args.overwrite:
                skipped += 1
                continue

            # Generate organ split
            success = generate_organ_split(
                organ_seq_path,
                plant_sequence_path,
                plant_split,
                output_dir,
            )

            if success:
                successful += 1
            else:
                failed += 1

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total organs processed: {total_organs}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped (already exist): {skipped}")
    print()
    print(f"Organ split files saved to: {output_dir}")


if __name__ == "__main__":
    main()
