"""
Extract stem sequences from plant sequences.

This script extracts stem (organ_label=0) point clouds from plant sequences
and saves them in the organ_sequences directory, similar to leaf sequences.

Usage:
    python scripts/extract_stem_sequences.py --dataset-path datasets/Pheno4D/processed/v2
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm


def extract_stem_timepoint(
    plant_ply_path: Path,
    output_path: Path,
) -> bool:
    """
    Extract stem (organ_label=0) from a plant timepoint and save as PLY.

    Args:
        plant_ply_path: Path to plant PLY file
        output_path: Output path for stem PLY file

    Returns:
        True if successful, False if no stem found
    """
    # Load plant point cloud
    pcd = o3d.t.io.read_point_cloud(str(plant_ply_path))
    points = pcd.point["positions"].numpy()

    # Check for organ labels
    if "organ_label" not in pcd.point:
        print(f"Warning: No organ_label in {plant_ply_path.name}")
        return False

    labels = pcd.point["organ_label"].numpy().flatten()

    # Find stem points (organ_label=0)
    stem_mask = labels == 0
    stem_indices = np.where(stem_mask)[0]

    if len(stem_indices) == 0:
        print(f"Warning: No stem points in {plant_ply_path.name}")
        return False

    # Extract stem data
    stem_points = points[stem_indices]

    # Create stem point cloud
    stem_pcd = o3d.t.geometry.PointCloud(
        o3d.core.Tensor(stem_points, dtype=o3d.core.Dtype.Float32)
    )

    # Copy normals if available
    if "normals" in pcd.point:
        normals = pcd.point["normals"].numpy()
        stem_normals = normals[stem_indices]
        stem_pcd.point["normals"] = o3d.core.Tensor(
            stem_normals, dtype=o3d.core.Dtype.Float32
        )

    # Note: We don't include leaf tips for stems (stems don't have tips)

    # Save stem point cloud
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.t.io.write_point_cloud(str(output_path), stem_pcd)

    return True


def extract_stem_sequence(
    plant_sequence_path: Path,
    output_dir: Path,
) -> bool:
    """
    Extract stem sequence from a plant sequence.

    Args:
        plant_sequence_path: Path to plant sequence
        output_dir: Output directory for stem sequence

    Returns:
        True if successful
    """
    plant_sparse_dir = plant_sequence_path / "sparse_pc"

    if not plant_sparse_dir.exists():
        print(f"Warning: {plant_sequence_path.name} has no sparse_pc directory")
        return False

    # Get all plant timepoint files
    plant_files = sorted(plant_sparse_dir.glob("*_sparse.ply"))

    if len(plant_files) == 0:
        print(f"Warning: No sparse PLY files in {plant_sequence_path.name}")
        return False

    # Create output directory
    stem_sparse_dir = output_dir / "sparse_pc"
    stem_sparse_dir.mkdir(parents=True, exist_ok=True)

    # Extract stem from each timepoint
    success_count = 0
    for plant_file in plant_files:
        # Create output filename by replacing plant name with stem name
        # E.g., "1_maize_control_plant1_D00_sparse.ply" -> "1_maize_control_plant1_D00_stem_sparse.ply"
        stem_filename = plant_file.stem.replace("_sparse", "_stem_sparse") + ".ply"
        stem_path = stem_sparse_dir / stem_filename

        # Extract stem
        if extract_stem_timepoint(plant_file, stem_path):
            success_count += 1

    if success_count == 0:
        print(f"Warning: No stem timepoints extracted for {plant_sequence_path.name}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract stem sequences from plant sequences"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/Pheno4D/processed/v2",
        help="Path to processed Pheno4D dataset (default: datasets/Pheno4D/processed/v2)",
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
        help="Overwrite existing stem sequences",
    )
    parser.add_argument(
        "--plants",
        type=str,
        nargs="+",
        help="Process only specific plants (e.g., maize_control_plant1)",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    plant_sequences_dir = dataset_path / "plant_sequences"
    organ_sequences_dir = dataset_path / "organ_sequences"

    print("=" * 80)
    print("Extracting stem sequences from plant sequences")
    print("=" * 80)
    print(f"Dataset path: {dataset_path}")
    print(f"Plant sequences: {plant_sequences_dir}")
    print(f"Organ sequences: {organ_sequences_dir}")
    print()

    # Find all plant sequences
    all_crops = ["maize", "sorghum", "tobacco", "tomato"]
    if args.crops:
        all_crops = args.crops

    total_plants = 0
    total_stems = 0
    successful = 0
    failed = 0
    skipped = 0

    for crop in all_crops:
        crop_plant_dir = plant_sequences_dir / crop
        crop_organ_dir = organ_sequences_dir / crop

        if not crop_plant_dir.exists():
            print(f"Skipping {crop} - no plant sequences found")
            continue

        # Get all plant sequences for this crop
        plant_sequences = sorted([p for p in crop_plant_dir.iterdir() if p.is_dir()])

        # Filter by specific plants if requested
        if args.plants:
            plant_sequences = [p for p in plant_sequences if p.name in args.plants]

        if len(plant_sequences) == 0:
            continue

        print(f"\nProcessing {crop}: {len(plant_sequences)} plant sequences")

        for plant_seq_path in tqdm(plant_sequences, desc=f"  {crop}"):
            total_plants += 1
            plant_name = plant_seq_path.name

            # Create stem sequence name
            stem_name = f"{plant_name}_stem"
            stem_output_dir = crop_organ_dir / stem_name

            # Check if already exists
            if stem_output_dir.exists() and not args.overwrite:
                skipped += 1
                continue

            # Clean directory if overwriting
            if stem_output_dir.exists() and args.overwrite:
                shutil.rmtree(stem_output_dir)

            # Extract stem sequence
            total_stems += 1
            success = extract_stem_sequence(plant_seq_path, stem_output_dir)

            if success:
                successful += 1
            else:
                failed += 1
                # Remove failed directory
                if stem_output_dir.exists():
                    shutil.rmtree(stem_output_dir)

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Plants processed: {total_plants}")
    print(f"Stem sequences:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped (already exist): {skipped}")
    print()
    print(f"Stem sequences saved to: {organ_sequences_dir}")


if __name__ == "__main__":
    main()
