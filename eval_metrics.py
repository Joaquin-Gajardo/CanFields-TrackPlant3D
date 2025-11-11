#!/usr/bin/env python3
"""
Evaluation script for computing quantitative metrics on 4D plant reconstructions.

Computes:
- Chamfer Distance (CD) between predicted and ground truth meshes for all timesteps
- End Point Error (EPE) for leaf tips
- Saves results to JSON files in the experiment directory
"""

import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import open3d as o3d
import point_cloud_utils as pcu
import torch
import trimesh

from metrics.metrics import PlantMetrics
from utils import add_common_args, configure_logging


def build_args():
    arg_parser = argparse.ArgumentParser(
        description="Compute quantitative metrics for 4D plant reconstructions"
    )
    arg_parser.add_argument(
        "--base_results_dir",
        default="results",
        required=False,
        help="The base results directory",
    )
    arg_parser.add_argument(
        "--methods",
        "-me",
        default="ode_solve",
        dest="methods",
        required=False,
        help="method name (default: ode_solve). Other options: 4DSDF,TPSR,NVFI,4DGF, ode_solve_woAdaptor",
    )
    arg_parser.add_argument(
        "--object_name",
        "-o",
        dest="object_name",
        required=True,
        help="Path to the object sequence (e.g., Pheno4D/processed/v2/organ_sequences/tomato/tomato2_control_plant2)",
    )
    arg_parser.add_argument(
        "--experiment_name",
        "-en",
        dest="experiment_name",
        required=True,
        help="Name of the specific experiment folder",
    )
    arg_parser.add_argument(
        "--dataset_name",
        "-d",
        dest="dataset_name",
        required=True,
        help="Dataset name: 'animals', 'medical', 'defaust', or 'pheno4d'",
    )
    arg_parser.add_argument(
        "--epoch",
        "-e",
        dest="epoch",
        required=True,
        help="Epoch to evaluate (e.g., 15000, 'last', or 'all')",
    )
    arg_parser.add_argument(
        "--num_samples",
        "-ns",
        dest="num_samples",
        type=int,
        default=10000,
        help="Number of points to sample from each mesh for CD computation (default: 10000)",
    )
    arg_parser.add_argument(
        "--no_tip_epe",
        dest="compute_tip_epe",
        action="store_false",
        default=True,
        help="Disable End Point Error (EPE) computation for leaf tips (EPE is computed by default using leaf tips stored in ground truth PLY files)",
    )
    arg_parser.add_argument(
        "--point_split_file",
        dest="point_split_file",
        default=None,
        required=False,
        help="Path to JSON file with train/test point splits (optional, for pheno4d)",
    )

    add_common_args(arg_parser)
    args = arg_parser.parse_args()
    configure_logging(args)
    return args


def load_mesh_as_points(mesh_path: str, num_samples: int = 10000) -> np.ndarray:
    """
    Load a mesh and sample points from its surface.

    Args:
        mesh_path: Path to mesh file (.ply)
        num_samples: Number of points to sample

    Returns:
        Sampled points as numpy array of shape (num_samples, 3)
    """
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    # Load mesh
    mesh = trimesh.load(mesh_path)

    # Sample points uniformly from surface
    points, _ = trimesh.sample.sample_surface(mesh, num_samples)

    return points


def load_leaf_tips_from_ply(ply_path: str) -> tuple:
    """
    Load leaf tip indices and coordinates from a PLY file.

    Args:
        ply_path: Path to PLY file with is_leaf_tip scalar field

    Returns:
        Tuple of (leaf_tip_indices, leaf_tip_coordinates)
    """
    try:
        # Load with Open3D tensor to access scalar fields
        pcd = o3d.t.io.read_point_cloud(ply_path)
        points = pcd.point.positions.numpy()

        # Check if leaf tips are available
        if "is_leaf_tip" in pcd.point:
            is_leaf_tip = pcd.point.is_leaf_tip.numpy().flatten().astype(bool)
            leaf_tip_idxs = np.where(is_leaf_tip)[0]
            leaf_tip_coords = points[leaf_tip_idxs]
            return leaf_tip_idxs, leaf_tip_coords
        else:
            return np.array([]), np.array([])
    except Exception as e:
        logging.warning(f"Could not load leaf tips from {ply_path}: {e}")
        return np.array([]), np.array([])


def load_point_splits(split_file: str) -> dict:
    """
    Load pre-computed train/test point splits from JSON file.

    Args:
        split_file: Path to point split JSON file

    Returns:
        Dictionary with split information
    """
    if not os.path.exists(split_file):
        logging.warning(f"Point split file not found: {split_file}")
        return None

    with open(split_file, 'r') as f:
        splits = json.load(f)

    logging.info(f"Loaded point splits from {split_file}")
    logging.info(f"  Sequence: {splits['sequence_name']}")
    logging.info(f"  Timepoints: {splits['n_timepoints']}")
    logging.info(f"  Train ratio: {splits['train_ratio']}")

    return splits


def apply_point_split(points: np.ndarray, split_indices: np.ndarray) -> np.ndarray:
    """
    Apply point split indices to a point cloud.

    Args:
        points: Point cloud array of shape (N, 3)
        split_indices: Indices to select from points

    Returns:
        Filtered point cloud
    """
    # Ensure indices are within bounds
    valid_indices = split_indices[split_indices < len(points)]
    if len(valid_indices) < len(split_indices):
        logging.warning(
            f"Some split indices out of bounds: {len(split_indices) - len(valid_indices)} indices removed"
        )
    return points[valid_indices]


def get_experiment_directory(args):
    """Build the experiment directory path based on dataset type."""
    if args.dataset_name == "pheno4d":
        path_parts = args.object_name.split("/")
        crop = path_parts[-2]  # Extract crop (e.g., "tomato")
        sequence_name = path_parts[-1]  # Extract sequence name
        experiment_folder = args.experiment_name.split("/")[-1]

        # Build path: results/pheno4d/{method}/{crop}/{sequence_name}/{experiment_name}/
        experiment_directory = os.path.join(
            args.base_results_dir,
            args.dataset_name,
            args.methods,
            crop,
            sequence_name,
            experiment_folder,
        )
    else:
        # Original structure for other datasets
        object_base_name = args.object_name.split("/")[-1]
        experiment_folder = args.experiment_name.split("/")[-1]

        experiment_directory = os.path.join(
            args.base_results_dir,
            args.dataset_name,
            object_base_name,
            args.methods,
            experiment_folder,
        )

    return experiment_directory


def get_ground_truth_paths(args):
    """Get paths to ground truth point clouds."""
    if args.dataset_name == "pheno4d":
        # Ground truth is in the sparse_pc or dense_pc folder
        pc_folder = "sparse_pc"
        pc_path = os.path.join(args.object_name, pc_folder)

        if not os.path.exists(pc_path):
            raise ValueError(f"Ground truth path does not exist: {pc_path}")

        # Get all .ply files and sort them
        gt_files = sorted([f for f in os.listdir(pc_path) if f.endswith(".ply")])

        # Extract parameters from experiment name to match the same filtering
        parts = args.experiment_name.split("_")
        start_idx = 0
        gap = 8
        stride = 1

        for i, part in enumerate(parts):
            if part == "start" and i + 2 < len(parts) and parts[i + 1] == "id":
                start_idx = int(parts[i + 2])
            elif part == "gap" and i + 1 < len(parts):
                gap = int(parts[i + 1])
            elif part == "stride" and i + 1 < len(parts):
                stride = int(parts[i + 1])

        # Apply same filtering as dataset
        end_idx = min(start_idx + gap, len(gt_files))
        gt_files = gt_files[start_idx:end_idx:stride]

        gt_paths = [os.path.join(pc_path, f) for f in gt_files]
        return gt_paths, gt_files
    else:
        raise NotImplementedError(
            f"Ground truth loading not implemented for {args.dataset_name}"
        )


def compute_metrics_for_epoch(args, epoch_num):
    """
    Compute metrics for a specific epoch.

    Args:
        args: Command line arguments
        epoch_num: Epoch number to evaluate

    Returns:
        Dictionary containing all computed metrics
    """
    logging.info(f"Computing metrics for epoch {epoch_num}")

    # Get experiment directory
    experiment_directory = get_experiment_directory(args)

    # Get predicted mesh directory
    pred_mesh_dir = os.path.join(
        experiment_directory,
        "manifold",
        f"epoch_{epoch_num}",
    )

    if not os.path.exists(pred_mesh_dir):
        raise FileNotFoundError(f"Predicted mesh directory not found: {pred_mesh_dir}")

    # Load point splits if provided (or auto-detect for pheno4d)
    point_splits = None
    split_file = args.point_split_file
    if args.dataset_name == "pheno4d" and split_file is None:
        # Auto-construct split file path
        sequence_name = args.object_name.split("/")[-1]
        split_file = f"configs/point_splits/{sequence_name}_split.json"

    if split_file:
        point_splits = load_point_splits(split_file)

    # Get ground truth paths
    gt_paths, gt_files = get_ground_truth_paths(args)

    # Get predicted mesh paths
    pred_files = [f for f in os.listdir(pred_mesh_dir) if f.endswith(".ply")]
    # Filter out normalized template
    pred_files = [f for f in pred_files if f != "normalized_tempt.ply"]
    pred_files = sorted(pred_files)

    if len(pred_files) == 0:
        raise ValueError(f"No predicted meshes found in {pred_mesh_dir}")

    logging.info(f"Found {len(pred_files)} predicted meshes")
    logging.info(f"Found {len(gt_files)} ground truth point clouds")

    if len(pred_files) != len(gt_files):
        logging.warning(
            f"Number of predicted meshes ({len(pred_files)}) does not match "
            f"number of ground truth point clouds ({len(gt_files)})"
        )
        # Use minimum length
        min_len = min(len(pred_files), len(gt_files))
        pred_files = pred_files[:min_len]
        gt_files = gt_files[:min_len]
        gt_paths = gt_paths[:min_len]

    # Initialize metrics
    metrics_calculator = PlantMetrics()

    # If point splits are provided, maintain separate train/test lists
    if point_splits:
        pred_points_train_list = []
        gt_points_train_list = []
        pred_points_test_list = []
        gt_points_test_list = []
        pred_tips_train_list = []
        gt_tips_train_list = []
        pred_tips_test_list = []
        gt_tips_test_list = []
    else:
        pred_points_list = []
        gt_points_list = []
        pred_tips_list = []
        gt_tips_list = []

    # Load all meshes and compute per-timestep metrics
    for i, (pred_file, gt_path) in enumerate(zip(pred_files, gt_paths)):
        pred_path = os.path.join(pred_mesh_dir, pred_file)

        logging.info(f"Processing timestep {i+1}/{len(pred_files)}: {pred_file}")

        # Load predicted mesh as points
        try:
            pred_points = load_mesh_as_points(pred_path, args.num_samples)
        except Exception as e:
            logging.error(f"Error loading predicted mesh {pred_file}: {e}")
            continue

        # Load ground truth point cloud
        try:
            gt_points, _, _ = pcu.load_mesh_vfn(gt_path, dtype=np.float32)
            # Deduplicate
            gt_points, _, _ = pcu.deduplicate_point_cloud(
                gt_points, 1e-15, return_index=True
            )
        except Exception as e:
            logging.error(f"Error loading ground truth {gt_path}: {e}")
            continue

        # Apply point splits if provided
        if point_splits:
            timepoint_data = point_splits['timepoints'][i]
            train_indices = np.array(timepoint_data['train_indices'], dtype=np.int64)
            test_indices = np.array(timepoint_data['test_indices'], dtype=np.int64)

            # Apply splits to GT points
            gt_points_train = apply_point_split(gt_points, train_indices)
            gt_points_test = apply_point_split(gt_points, test_indices)

            # For predicted points, we need to use the same split indices
            # Note: pred_points may have different number of points, so we need to be careful
            pred_points_train = apply_point_split(pred_points, train_indices)
            pred_points_test = apply_point_split(pred_points, test_indices)

            # Convert to tensors
            pred_points_train_tensor = torch.tensor(pred_points_train, dtype=torch.float32).cuda()
            gt_points_train_tensor = torch.tensor(gt_points_train, dtype=torch.float32).cuda()
            pred_points_test_tensor = torch.tensor(pred_points_test, dtype=torch.float32).cuda()
            gt_points_test_tensor = torch.tensor(gt_points_test, dtype=torch.float32).cuda()

            pred_points_train_list.append(pred_points_train_tensor)
            gt_points_train_list.append(gt_points_train_tensor)
            pred_points_test_list.append(pred_points_test_tensor)
            gt_points_test_list.append(gt_points_test_tensor)

            logging.info(f"  Train points: pred={len(pred_points_train)}, gt={len(gt_points_train)}")
            logging.info(f"  Test points: pred={len(pred_points_test)}, gt={len(gt_points_test)}")

            # Also create full pred_points_tensor for tip EPE computation (if needed)
            pred_points_tensor = torch.tensor(pred_points, dtype=torch.float32).cuda()
            gt_points_tensor = torch.tensor(gt_points, dtype=torch.float32).cuda()
        else:
            # No splits - use all points
            pred_points_tensor = torch.tensor(pred_points, dtype=torch.float32).cuda()
            gt_points_tensor = torch.tensor(gt_points, dtype=torch.float32).cuda()

            pred_points_list.append(pred_points_tensor)
            gt_points_list.append(gt_points_tensor)

        # Load leaf tips from ground truth PLY file if requested
        if args.compute_tip_epe:
            # Load GT leaf tips from PLY file
            _, gt_tip_coords = load_leaf_tips_from_ply(gt_path)

            if len(gt_tip_coords) > 0:
                # Convert to tensor
                gt_tips_tensor = torch.tensor(gt_tip_coords, dtype=torch.float32).cuda()

                # For predicted tips, find closest points in predicted mesh to GT tip positions
                # This assumes predicted and GT are in the same coordinate space
                pred_tips_tensor = torch.zeros_like(gt_tips_tensor)
                for j, gt_tip in enumerate(gt_tips_tensor):
                    # Find nearest point in predicted point cloud
                    distances = torch.norm(
                        pred_points_tensor - gt_tip.unsqueeze(0), dim=1
                    )
                    nearest_idx = torch.argmin(distances)
                    pred_tips_tensor[j] = pred_points_tensor[nearest_idx]

                pred_tips_list.append(pred_tips_tensor)
                gt_tips_list.append(gt_tips_tensor)
            else:
                logging.warning(
                    f"No leaf tips found in {gt_path}, skipping EPE for this timestep"
                )
                # Add None placeholders to maintain list alignment
                pred_tips_list.append(None)
                gt_tips_list.append(None)

    # Compute metrics across all timesteps
    metrics_json = {
        "epoch": epoch_num,
        "num_samples_per_mesh": args.num_samples,
        "has_point_splits": point_splits is not None,
    }

    if point_splits:
        # Compute separate train and test metrics
        logging.info("Computing separate train and test metrics...")

        # Train metrics
        metrics_train = metrics_calculator.compute_metrics_per_timestep(
            pred_points_train_list, gt_points_train_list
        )
        metrics_json["train"] = {
            "num_timesteps": len(pred_points_train_list),
            "mean_chamfer_distance": float(metrics_train["mean_chamfer"].cpu().item()),
            "std_chamfer_distance": float(metrics_train["std_chamfer"].cpu().item()),
            "chamfer_distances_per_timestep": [
                float(cd.cpu().item()) for cd in metrics_train["chamfer_distances"]
            ],
        }

        # Test metrics
        metrics_test = metrics_calculator.compute_metrics_per_timestep(
            pred_points_test_list, gt_points_test_list
        )
        metrics_json["test"] = {
            "num_timesteps": len(pred_points_test_list),
            "mean_chamfer_distance": float(metrics_test["mean_chamfer"].cpu().item()),
            "std_chamfer_distance": float(metrics_test["std_chamfer"].cpu().item()),
            "chamfer_distances_per_timestep": [
                float(cd.cpu().item()) for cd in metrics_test["chamfer_distances"]
            ],
        }

        # Log results
        logging.info(f"Train Set - Mean CD: {metrics_json['train']['mean_chamfer_distance']:.6f}")
        logging.info(f"Train Set - Std CD: {metrics_json['train']['std_chamfer_distance']:.6f}")
        logging.info(f"Test Set - Mean CD: {metrics_json['test']['mean_chamfer_distance']:.6f}")
        logging.info(f"Test Set - Std CD: {metrics_json['test']['std_chamfer_distance']:.6f}")

    else:
        # No splits - compute metrics on all points (backward compatible)
        if args.compute_tip_epe and len(pred_tips_list) > 0 and len(gt_tips_list) > 0:
            # Filter out None values (timesteps without leaf tips)
            valid_pred_tips = [t for t in pred_tips_list if t is not None]
            valid_gt_tips = [t for t in gt_tips_list if t is not None]

            if len(valid_pred_tips) > 0 and len(valid_gt_tips) > 0:
                metrics = metrics_calculator.compute_metrics_per_timestep(
                    pred_points_list, gt_points_list, valid_pred_tips, valid_gt_tips
                )
                logging.info(
                    f"Computing tip EPE for {len(valid_pred_tips)}/{len(pred_tips_list)} timesteps with leaf tips"
                )
            else:
                metrics = metrics_calculator.compute_metrics_per_timestep(
                    pred_points_list, gt_points_list
                )
                logging.warning("No valid leaf tips found, skipping tip EPE computation")
        else:
            metrics = metrics_calculator.compute_metrics_per_timestep(
                pred_points_list, gt_points_list
            )

        metrics_json["num_timesteps"] = len(pred_points_list)
        metrics_json["mean_chamfer_distance"] = float(metrics["mean_chamfer"].cpu().item())
        metrics_json["std_chamfer_distance"] = float(metrics["std_chamfer"].cpu().item())
        metrics_json["chamfer_distances_per_timestep"] = [
            float(cd.cpu().item()) for cd in metrics["chamfer_distances"]
        ]

        # Compute tip EPE if requested and tips were found
        if args.compute_tip_epe and "mean_epe" in metrics:
            # Count valid tips (filter out None)
            valid_tips_count = sum(1 for t in pred_tips_list if t is not None)
            metrics_json["num_timesteps_with_tips"] = valid_tips_count
            metrics_json["mean_tip_epe"] = float(metrics["mean_epe"].cpu().item())
            metrics_json["std_tip_epe"] = float(metrics["std_epe"].cpu().item())
            metrics_json["tip_epe_per_timestep"] = [
                float(epe.cpu().item()) for epe in metrics["epe"]
            ]
        elif args.compute_tip_epe:
            logging.warning("Tip EPE requested but no leaf tips were found in any timestep")

        # Log results
        logging.info(f"Mean Chamfer Distance: {metrics_json['mean_chamfer_distance']:.6f}")
        logging.info(f"Std Chamfer Distance: {metrics_json['std_chamfer_distance']:.6f}")
        if args.compute_tip_epe and "mean_tip_epe" in metrics_json:
            logging.info(f"Mean Tip EPE: {metrics_json['mean_tip_epe']:.6f}")
            logging.info(f"Std Tip EPE: {metrics_json['std_tip_epe']:.6f}")

    return metrics_json


def save_metrics_to_json(metrics, experiment_directory, epoch_num):
    """Save metrics to JSON file in experiment directory."""
    metrics_dir = os.path.join(experiment_directory, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Save per-epoch metrics
    metrics_file = os.path.join(metrics_dir, f"metrics_epoch_{epoch_num}.json")

    # Add timestamp
    metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics["experiment_directory"] = experiment_directory

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info(f"Metrics saved to: {metrics_file}")

    return metrics_file


def main():
    args = build_args()

    experiment_directory = get_experiment_directory(args)

    if not os.path.exists(experiment_directory):
        raise FileNotFoundError(
            f"Experiment directory not found: {experiment_directory}"
        )

    logging.info(f"Evaluating experiment: {experiment_directory}")

    # Handle epoch selection
    if args.epoch.lower() == "all":
        # Find all available epochs
        manifold_dir = os.path.join(experiment_directory, "manifold")
        if not os.path.exists(manifold_dir):
            raise FileNotFoundError(f"Manifold directory not found: {manifold_dir}")

        epoch_dirs = [d for d in os.listdir(manifold_dir) if d.startswith("epoch_")]
        epochs = [int(d.split("_")[1]) for d in epoch_dirs]
        epochs = sorted(epochs)
    elif args.epoch.lower() == "last":
        # Find the last epoch
        manifold_dir = os.path.join(experiment_directory, "manifold")
        if not os.path.exists(manifold_dir):
            raise FileNotFoundError(f"Manifold directory not found: {manifold_dir}")

        epoch_dirs = [d for d in os.listdir(manifold_dir) if d.startswith("epoch_")]
        epochs = [int(d.split("_")[1]) for d in epoch_dirs]
        epochs = [max(epochs)]
    else:
        epochs = [int(args.epoch)]

    logging.info(f"Evaluating {len(epochs)} epoch(s): {epochs}")

    # Compute metrics for each epoch
    all_metrics = []
    failed_epochs = []
    for epoch_num in epochs:
        try:
            metrics = compute_metrics_for_epoch(args, epoch_num)
            save_metrics_to_json(metrics, experiment_directory, epoch_num)
            all_metrics.append(metrics)
        except Exception as e:
            logging.error(f"Error computing metrics for epoch {epoch_num}: {e}")
            failed_epochs.append(epoch_num)
            continue

    # Save summary if multiple epochs
    if len(all_metrics) > 1:
        summary = {
            "num_epochs_evaluated": len(all_metrics),
            "epochs": [m["epoch"] for m in all_metrics],
            "mean_chamfer_distances": [m["mean_chamfer_distance"] for m in all_metrics],
            "mean_epe": [m.get("mean_tip_epe", None) for m in all_metrics],
            "best_epoch_by_chamfer": min(
                all_metrics, key=lambda x: x["mean_chamfer_distance"]
            )["epoch"],
        }

        if args.compute_tip_epe:
            valid_epe_metrics = [m for m in all_metrics if "mean_tip_epe" in m]
            if valid_epe_metrics:
                summary["best_epoch_by_tip_epe"] = min(
                    valid_epe_metrics, key=lambda x: x["mean_tip_epe"]
                )["epoch"]

        summary_file = os.path.join(
            experiment_directory, "metrics", "metrics_summary.json"
        )
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logging.info(f"Summary saved to: {summary_file}")

    logging.info(
        f"Evaluation complete! Processed {len(all_metrics)} epoch(s) successfully, {len(failed_epochs)} failed"
    )
    if failed_epochs:
        logging.warning(f"Failed epochs: {failed_epochs}")


if __name__ == "__main__":
    main()
