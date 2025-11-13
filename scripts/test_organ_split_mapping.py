"""
Test script to verify organ split mapping logic works correctly.

Tests on a single organ sequence to ensure plant indices map correctly to organ indices.
"""

import json
from pathlib import Path

import numpy as np
import open3d as o3d


def load_point_cloud(ply_path):
    """Load point cloud and extract points and organ labels."""
    pcd = o3d.t.io.read_point_cloud(str(ply_path))
    points = pcd.point["positions"].numpy()

    labels = None
    if "organ_label" in pcd.point:
        labels = pcd.point["organ_label"].numpy().flatten()

    return points, labels


def get_organ_id_from_filename(filename):
    """Extract organ ID from filename."""
    import re

    # Check for leaf pattern (e.g., 'day_0_leaf3_sparse.ply')
    leaf_match = re.search(r'_leaf(\d+)_', filename)
    if leaf_match:
        return int(leaf_match.group(1))

    # Check for stem pattern
    if '_stem_' in filename:
        return 0

    return None


def test_mapping():
    """Test organ mapping on maize_control_plant1_leaf1."""

    # Paths
    plant_split_file = Path("configs/point_splits/maize_control_plant1_split.json")
    plant_seq_dir = Path("datasets/Pheno4D/processed/v2/plant_sequences/maize/maize_control_plant1")
    organ_seq_dir = Path("datasets/Pheno4D/processed/v2/organ_sequences/maize/maize_control_plant1_leaf1")

    print("=" * 80)
    print("Testing organ split mapping")
    print("=" * 80)
    print(f"Plant: {plant_seq_dir.name}")
    print(f"Organ: {organ_seq_dir.name}")
    print()

    # Load plant split
    with open(plant_split_file) as f:
        plant_split = json.load(f)

    print(f"Plant split info:")
    print(f"  Random seed: {plant_split['random_seed']}")
    print(f"  Train ratio: {plant_split['train_ratio']}")
    print(f"  Timepoints: {plant_split['n_timepoints']}")
    print()

    # Test on first timepoint
    tp_idx = 0
    plant_tp_split = plant_split["timepoints"][tp_idx]

    # Load point clouds
    plant_files = sorted((plant_seq_dir / "sparse_pc").glob("*_sparse.ply"))
    organ_files = sorted((organ_seq_dir / "sparse_pc").glob("*_sparse.ply"))

    plant_file = plant_files[tp_idx]
    organ_file = organ_files[tp_idx]

    print(f"Loading timepoint {tp_idx}:")
    print(f"  Plant file: {plant_file.name}")
    print(f"  Organ file: {organ_file.name}")
    print()

    # Load data
    plant_points, plant_labels = load_point_cloud(plant_file)
    organ_points, _ = load_point_cloud(organ_file)

    print(f"Point cloud sizes:")
    print(f"  Plant: {len(plant_points)} points")
    print(f"  Organ: {len(organ_points)} points")
    print()

    # Extract organ ID from filename
    organ_id = get_organ_id_from_filename(organ_file.name)
    print(f"Detected organ ID: {organ_id}")
    print()

    # Check if plant has organ labels
    if plant_labels is None:
        print("ERROR: Plant point cloud doesn't have organ_label field!")
        return False

    print(f"Plant organ labels:")
    unique_labels, counts = np.unique(plant_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        label_str = "stem" if label == 0 else f"leaf{label}"
        print(f"  Organ {label} ({label_str}): {count} points")
    print()

    # Find which plant points belong to this organ
    organ_mask = plant_labels == organ_id
    organ_plant_indices = np.where(organ_mask)[0]
    organ_points_from_plant = plant_points[organ_plant_indices]

    print(f"Points belonging to organ {organ_id} in plant: {len(organ_plant_indices)}")
    print(f"Points in organ sequence: {len(organ_points)}")

    # Check if sizes match
    if len(organ_plant_indices) != len(organ_points):
        print(f"WARNING: Size mismatch! This could mean:")
        print(f"  - Organ sequence was filtered/preprocessed differently")
        print(f"  - Need to use coordinate matching instead of sequential mapping")
        print()

    # Build mapping from plant index to organ index
    plant_to_organ_map = {}
    for organ_idx, plant_idx in enumerate(organ_plant_indices):
        plant_to_organ_map[plant_idx] = organ_idx

    # Get plant train/test indices
    plant_train_indices = plant_tp_split["train_indices"]
    plant_test_indices = plant_tp_split["test_indices"]

    print(f"\nPlant split for timepoint {tp_idx}:")
    print(f"  Total points: {plant_tp_split['n_total_points']}")
    print(f"  Train points: {plant_tp_split['n_train']} ({plant_tp_split['n_train']/plant_tp_split['n_total_points']*100:.1f}%)")
    print(f"  Test points: {plant_tp_split['n_test']} ({plant_tp_split['n_test']/plant_tp_split['n_total_points']*100:.1f}%)")
    print()

    # Map to organ indices
    organ_train_indices = []
    organ_test_indices = []

    for plant_idx in plant_train_indices:
        if plant_idx in plant_to_organ_map:
            organ_train_indices.append(plant_to_organ_map[plant_idx])

    for plant_idx in plant_test_indices:
        if plant_idx in plant_to_organ_map:
            organ_test_indices.append(plant_to_organ_map[plant_idx])

    print(f"Organ split for timepoint {tp_idx}:")
    print(f"  Total points: {len(organ_points)}")
    print(f"  Train points: {len(organ_train_indices)} ({len(organ_train_indices)/len(organ_points)*100:.1f}%)")
    print(f"  Test points: {len(organ_test_indices)} ({len(organ_test_indices)/len(organ_points)*100:.1f}%)")
    print()

    # Verify coverage
    mapped_points = len(organ_train_indices) + len(organ_test_indices)
    coverage = mapped_points / len(organ_points) * 100

    print(f"Mapping coverage: {mapped_points}/{len(organ_points)} points ({coverage:.1f}%)")

    if coverage < 95:
        print("WARNING: Low coverage! Some organ points weren't mapped.")
    else:
        print("✓ Good coverage - mapping is working correctly")

    print()
    print("=" * 80)
    print("Test complete!")
    print("=" * 80)

    return True


if __name__ == "__main__":
    test_mapping()
