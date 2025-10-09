"""
Prepare Pheno4D sequences for CanFields training.

This script uses plant-shape-analysis dataloaders to automatically export
plant or leaf sequences in the format expected by CanFields training code.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
from plant_shape_analysis import LeafSequencesDataset, PlantSequencesDataset
from tqdm import tqdm


def export_timepoint(
    points: np.ndarray,
    output_path: Path,
    estimate_normals: bool = True,
):
    """
    Export a single timepoint as PLY file.

    Args:
        points: Point cloud array (N, 3) or (N, 6) with normals
        output_path: Output file path
        estimate_normals: If True, estimate normals with Open3D
    """
    if points is None or len(points) == 0:
        return False

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # x, y, z

    # Handle normals
    if estimate_normals:
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=30)
        pcd.orient_normals_to_align_with_direction()
    elif points.shape[1] == 6:
        # Normals already in the data
        pcd.normals = o3d.utility.Vector3dVector(points[:, 3:6])

    # Save point cloud
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), pcd)
    return True


def export_leaf_sequence(
    leaf_timeseries: dict,
    output_dir: Path,
    export_dense: bool = True,
    export_sparse: bool = True,
    estimate_normals: bool = True,
):
    """
    Export a single leaf sequence to CanFields format.

    Args:
        leaf_timeseries: Leaf timeseries dict from LeafSequencesDataset
        output_dir: Output directory for this sequence
        export_dense: If True, export dense point clouds
        export_sparse: If True, export sparse point clouds
        estimate_normals: If True, estimate and include normals
    """
    sequence_name = leaf_timeseries["sequence_name"]
    timepoints = leaf_timeseries["timepoints"]

    # Create mesh directory
    mesh_dir = output_dir / "gt_mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    for tp in timepoints:
        day = tp["day"]
        leaf_id = tp["leaf_id"]
        file_stem = tp["file_path"].stem

        # Export dense if requested and available
        if export_dense and tp.get("dense_points") is not None:
            filename = f"{file_stem}_leaf{leaf_id}_dense.ply"
            dense_path = output_dir / "dense_pc" / filename
            export_timepoint(tp["dense_points"], dense_path, estimate_normals)

        # Export sparse if requested
        if export_sparse and tp.get("points") is not None:
            filename = f"{file_stem}_leaf{leaf_id}_sparse.ply"
            sparse_path = output_dir / "sparse_pc" / filename
            export_timepoint(tp["points"], sparse_path, estimate_normals)


def export_plant_sequence(
    plant_timeseries: dict,
    output_dir: Path,
    export_dense: bool = True,
    export_sparse: bool = True,
    estimate_normals: bool = True,
):
    """
    Export a single plant sequence to CanFields format.

    Args:
        plant_timeseries: Plant timeseries dict from PlantSequencesDataset
        output_dir: Output directory for this sequence
        export_dense: If True, export dense point clouds
        export_sparse: If True, export sparse point clouds
        estimate_normals: If True, estimate and include normals
    """
    sequence_name = plant_timeseries["sequence_name"]
    timepoints = plant_timeseries["timepoints"]

    # Create mesh directory
    mesh_dir = output_dir / "gt_mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    for tp in timepoints:
        day = tp["day"]
        file_stem = tp["file_path"].stem

        # Export dense if requested and available
        if export_dense and tp.get("dense_points") is not None:
            filename = f"{file_stem}_dense.ply"
            dense_path = output_dir / "dense_pc" / filename
            export_timepoint(tp["dense_points"], dense_path, estimate_normals)

        # Export sparse if requested
        if export_sparse and tp.get("points") is not None:
            filename = f"{file_stem}_sparse.ply"
            sparse_path = output_dir / "sparse_pc" / filename
            export_timepoint(tp["points"], sparse_path, estimate_normals)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Pheno4D sequences for CanFields training"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/Pheno4D/raw",
        help="Path to TrackPlant3D dataset (will auto-download if not found, default: datasets/Pheno4D/raw)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Dataset version to use (default: v1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/Pheno4D/processed",
        help="Output directory for CanFields-format sequences (default: datasets/Pheno4D/processed)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["leaf", "plant", "both"],
        default="both",
        help="Export mode: 'leaf' (individual leaves), 'plant' (whole plants), or 'both' (default: leaf)",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        help="Specific sequences to export (e.g., maize_control_plant2_leaf2 or maize_control_plant2). If not specified, export all.",
    )
    parser.add_argument(
        "--min-timepoints",
        type=int,
        default=5,
        help="Minimum number of timepoints required (default: 5)",
    )
    parser.add_argument(
        "--export-dense",
        action="store_true",
        default=False,
        help="Export dense point clouds (default: True)",
    )
    parser.add_argument(
        "--no-dense",
        dest="export_dense",
        action="store_false",
        help="Don't export dense point clouds",
    )
    parser.add_argument(
        "--export-sparse",
        action="store_true",
        default=True,
        help="Export sparse point clouds (default: True)",
    )
    parser.add_argument(
        "--no-sparse",
        dest="export_sparse",
        action="store_false",
        help="Don't export sparse point clouds",
    )
    parser.add_argument(
        "--estimate-normals",
        action="store_true",
        default=True,
        help="Estimate normals for point clouds (default: True)",
    )
    parser.add_argument(
        "--crop",
        type=str,
        help="Filter by crop (e.g., maize, tomato2)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sequences",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Preparing Pheno4D sequences for CanFields")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Export dense: {args.export_dense}")
    print(f"Export sparse: {args.export_sparse}")

    dataset_path = Path(args.dataset_path)
    base_output_dir = Path(args.output_dir) / args.version

    # Determine which datasets to load
    load_plant = args.mode in ["plant", "both"]
    load_leaf = args.mode in ["leaf", "both"]

    all_sequences = []

    # Load plant sequences if requested
    if load_plant:
        print(
            f"\nLoading PlantSequencesDataset from {dataset_path} (version: {args.version})..."
        )
        plant_dataset = PlantSequencesDataset(
            dataset_path=dataset_path,
            version=args.version,
            use_ply=True,
            alignment_method=None,
            auto_download=True,
        )

        # Filter by min_timepoints
        plant_sequences = [
            ts
            for ts in plant_dataset.plant_timeseries
            if len(ts["timepoints"]) >= args.min_timepoints
        ]
        print(
            f"✓ Loaded {len(plant_sequences)} plant sequences (with {args.min_timepoints}+ timepoints)"
        )

        # Mark as plant sequences
        for ts in plant_sequences:
            ts["_type"] = "plant"
        all_sequences.extend(plant_sequences)

    # Load leaf sequences if requested
    if load_leaf:
        print(
            f"\nLoading LeafSequencesDataset from {dataset_path} (version: {args.version})..."
        )
        leaf_dataset = LeafSequencesDataset(
            dataset_path=dataset_path,
            version=args.version,
            min_timepoints=args.min_timepoints,
            use_ply=True,
            alignment_method=None,
            estimate_normals=False,
            auto_download=True,
        )
        print(
            f"✓ Loaded {len(leaf_dataset)} leaf sequences (with {args.min_timepoints}+ timepoints)"
        )

        # Mark as leaf sequences
        for ts in leaf_dataset.leaf_timeseries:
            ts["_type"] = "leaf"
        all_sequences.extend(leaf_dataset.leaf_timeseries)

    # Filter sequences if specified
    if args.sequences:
        all_sequences = [
            ts for ts in all_sequences if ts["sequence_name"] in args.sequences
        ]
        print(f"Filtering to {len(all_sequences)} specified sequences")

    if args.crop:
        all_sequences = [
            ts for ts in all_sequences if ts["sequence_name"].startswith(args.crop)
        ]
        print(f"Filtering to {len(all_sequences)} sequences from crop: {args.crop}")

    if len(all_sequences) == 0:
        print("No sequences to export!")
        return

    # Group by crop for organized output
    crop_sequences = {}
    for ts in all_sequences:
        crop_name = ts["sequence_name"].split("_")[0].removesuffix("2")  # for tomato2
        if crop_name not in crop_sequences:
            crop_sequences[crop_name] = []
        crop_sequences[crop_name].append(ts)

    print(f"\nExporting {len(all_sequences)} sequences grouped by crop:")
    for crop, seqs in crop_sequences.items():
        plant_count = sum(1 for s in seqs if s["_type"] == "plant")
        leaf_count = sum(1 for s in seqs if s["_type"] == "leaf")
        print(
            f"  {crop}: {len(seqs)} sequences ({plant_count} plants, {leaf_count} leaves)"
        )

    # Export sequences
    print("\nExporting sequences...")
    for crop_name, sequences in crop_sequences.items():
        for ts in tqdm(sequences, desc=f"Exporting {crop_name}"):
            sequence_name = ts["sequence_name"]

            # Determine output directory based on sequence type
            if ts["_type"] == "plant":
                seq_output_dir = (
                    base_output_dir / "plant_sequences" / crop_name / sequence_name
                )
            else:  # leaf
                seq_output_dir = (
                    base_output_dir / "organ_sequences" / crop_name / sequence_name
                )

            # Check if already exists
            if seq_output_dir.exists() and not args.overwrite:
                continue

            # Clean directory if overwriting
            if seq_output_dir.exists() and args.overwrite:
                shutil.rmtree(seq_output_dir)

            # Export based on type
            if ts["_type"] == "plant":
                export_plant_sequence(
                    ts,
                    seq_output_dir,
                    export_dense=args.export_dense,
                    export_sparse=args.export_sparse,
                    estimate_normals=args.estimate_normals,
                )
            else:  # leaf
                export_leaf_sequence(
                    ts,
                    seq_output_dir,
                    export_dense=args.export_dense,
                    export_sparse=args.export_sparse,
                    estimate_normals=args.estimate_normals,
                )

    print("\n" + "=" * 80)
    print("✓ Export complete!")
    print("=" * 80)

    # Show where sequences were exported
    if args.mode == "leaf":
        print(f"\nOrgan sequences exported to: {base_output_dir}/organ_sequences/")
        example_prefix = "organ_sequences"
    elif args.mode == "plant":
        print(f"\nPlant sequences exported to: {base_output_dir}/plant_sequences/")
        example_prefix = "plant_sequences"
    else:  # both
        print(f"\nOrgan sequences exported to: {base_output_dir}/organ_sequences/")
        print(f"Plant sequences exported to: {base_output_dir}/plant_sequences/")
        example_prefix = "organ_sequences"

    print("\nExample usage in CanFields training:")
    example_seq = all_sequences[0]["sequence_name"]
    example_crop = example_seq.split("_")[0]
    if example_crop.startswith("tomato"):
        example_crop = "tomato"
    print(
        f'  python train.py --object_name "Pheno4D/processed/{args.version}/{example_prefix}/{example_crop}/{example_seq}"'
    )


if __name__ == "__main__":
    main()
