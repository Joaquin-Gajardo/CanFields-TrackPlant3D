#!/usr/bin/env python3
"""
Evaluation script for computing quantitative metrics on 4D plant reconstructions.

Computes:
- Chamfer Distance (CD) using PyTorch3D's efficient implementation
  - Bidirectional squared L2 distance between predicted and GT point clouds
- Normal Consistency metric
  - Measures alignment of surface normals between predicted and GT meshes
  - Score of 1 = perfect alignment, 0 = orthogonal
- End Point Error (EPE) for leaf tips using warper-based tracking
  - Tracks tips through time using the learned deformation field
  - Handles new tips appearing at later timepoints
  - Much more accurate than nearest-neighbor matching
- Saves results to JSON files in the experiment directory

Supports train/test splits for separate evaluation on different point subsets.
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
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points

from metrics.metrics import PlantMetrics
from utils import add_common_args, configure_logging, load_latent_vectors, load_model_parameters


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
    arg_parser.add_argument(
        "--specifications",
        "-s",
        dest="specifications",
        default="configs/specs.json",
        help="Path to specs JSON file (required for warper-based tip tracking)",
    )
    arg_parser.add_argument(
        "--no_normal_consistency",
        dest="compute_normal_consistency",
        action="store_false",
        default=True,
        help="Disable Normal Consistency metric computation (enabled by default)",
    )

    add_common_args(arg_parser)
    args = arg_parser.parse_args()
    configure_logging(args)
    return args


def load_mesh_as_points(mesh_path: str, num_samples: int = 10000, return_normals: bool = False):
    """
    Load a mesh and sample points from its surface.

    Args:
        mesh_path: Path to mesh file (.ply)
        num_samples: Number of points to sample
        return_normals: If True, also return surface normals at sampled points

    Returns:
        If return_normals=False: Sampled points as numpy array of shape (num_samples, 3)
        If return_normals=True: Tuple of (points, normals) each of shape (num_samples, 3)
    """
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    # Load mesh
    mesh = trimesh.load(mesh_path)

    # Sample points uniformly from surface
    points, face_indices = trimesh.sample.sample_surface(mesh, num_samples)

    if return_normals:
        # Get normals at the sampled points
        # Use face normals for the faces that were sampled
        normals = mesh.face_normals[face_indices]
        return points, normals
    else:
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

    with open(split_file, "r") as f:
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


def compute_pytorch3d_chamfer_distance(pred_points: torch.Tensor, gt_points: torch.Tensor) -> float:
    """
    Compute Chamfer Distance using PyTorch3D's implementation.

    Args:
        pred_points: Predicted point cloud of shape (N, 3)
        gt_points: Ground truth point cloud of shape (M, 3)

    Returns:
        Chamfer distance (sum of bidirectional distances)
    """
    # PyTorch3D expects batch dimension
    pred_batch = pred_points.unsqueeze(0)  # (1, N, 3)
    gt_batch = gt_points.unsqueeze(0)      # (1, M, 3)

    # chamfer_distance returns (loss1, loss2) where:
    # loss1: pred->gt distance
    # loss2: gt->pred distance
    # Both are already squared L2 distances
    cd_loss, _ = chamfer_distance(pred_batch, gt_batch)

    return cd_loss.item()


def compute_normal_consistency(
    pred_points: torch.Tensor,
    pred_normals: torch.Tensor,
    gt_points: torch.Tensor,
    gt_normals: torch.Tensor
) -> float:
    """
    Compute Normal Consistency metric between predicted and GT point clouds.

    For each predicted point, find the nearest GT point and compute the
    absolute dot product between their normals. Average across all points.

    Args:
        pred_points: Predicted points of shape (N, 3)
        pred_normals: Predicted normals of shape (N, 3)
        gt_points: Ground truth points of shape (M, 3)
        gt_normals: Ground truth normals of shape (M, 3)

    Returns:
        Normal consistency score (1 = perfect alignment, 0 = orthogonal)
    """
    # Add batch dimension for knn_points
    pred_batch = pred_points.unsqueeze(0)  # (1, N, 3)
    gt_batch = gt_points.unsqueeze(0)      # (1, M, 3)

    # Find nearest GT point for each predicted point
    knn_result = knn_points(pred_batch, gt_batch, K=1)
    nearest_indices = knn_result.idx.squeeze(0).squeeze(-1)  # (N,)

    # Get normals at nearest GT points
    nearest_gt_normals = gt_normals[nearest_indices]  # (N, 3)

    # Compute dot product between pred and nearest GT normals
    dot_products = (pred_normals * nearest_gt_normals).sum(dim=-1)  # (N,)

    # Take absolute value and average
    normal_consistency = torch.abs(dot_products).mean()

    return normal_consistency.item()


def load_decoder_model(args, specs, sdf_dataset, epoch_num):
    """
    Load the trained decoder model for warper-based tip tracking.

    Args:
        args: Command line arguments
        specs: Specifications loaded from JSON
        sdf_dataset: Dataset object containing normalization parameters
        epoch_num: Epoch number to load

    Returns:
        Tuple of (decoder, scale, mean, normalized_times)
    """
    from networks.multi_velocity_debug import Decoder, adjust_net

    logging.info("Loading trained decoder model for tip tracking...")

    latent_size = specs["CodeLength"]

    # Create adjustment network
    ajust = adjust_net().cuda()

    # Create decoder
    decoder = Decoder(
        "ode_solve",
        latent_size,
        specs["NetworkSpecs"]["warper_kargs"],
        specs["NetworkSpecs"]["decoder_debug_kargs"],
    )
    decoder = decoder.cuda()

    # Create and equip embeddings
    start_end_vecs = torch.nn.Embedding(2, 128, max_norm=1.0).cuda()
    decoder.warper.equip_embedding(start_end_vecs)

    # Load model as ModuleList (adjust + decoder)
    experiment_directory = get_experiment_directory(args)
    epoch = epoch_num

    composed_model = torch.nn.ModuleList().cuda()
    composed_model.append(ajust)
    composed_model.append(decoder)

    load_model_parameters(experiment_directory, epoch, composed_model)

    # Extract decoder after loading
    ajust = composed_model[0]
    decoder = composed_model[1]

    load_latent_vectors(experiment_directory, epoch, start_end_vecs)

    decoder.eval()
    ajust.eval()

    # Set time range
    start_time = sdf_dataset.normalized_times[0]
    end_time = sdf_dataset.normalized_times[-1]
    decoder.warper.split_time(start_time, end_time)

    # Get normalization parameters
    scale = sdf_dataset.scale
    mean = sdf_dataset.mean

    logging.info("Decoder model loaded successfully")

    return decoder, scale, mean, sdf_dataset.normalized_times


def track_tips_with_warper(
    decoder,
    gt_paths,
    sdf_dataset,
    scale,
    mean,
    normalized_times,
    point_splits=None
):
    """
    Track leaf tips through time using the learned warper.

    This function:
    1. Identifies all leaf tips across all timepoints
    2. Tracks each tip from its first appearance through subsequent timepoints
    3. Handles new tips appearing at later timepoints

    Args:
        decoder: Trained decoder with warper
        gt_paths: List of paths to ground truth PLY files
        sdf_dataset: Dataset object
        scale: Normalization scale
        mean: Normalization mean
        normalized_times: Tensor of normalized time values for each frame
        point_splits: Optional dict with train/test splits

    Returns:
        Dictionary with tracked tips and EPE values per split
    """
    logging.info("Tracking leaf tips with warper...")

    # Dictionary to track tips: {tip_id: (first_appear_idx, first_appear_coords, first_appear_time)}
    tracked_tips = {}

    # Load all GT tips first to identify unique tips
    all_gt_tips = []
    for t_idx, gt_path in enumerate(gt_paths):
        _, gt_tip_coords = load_leaf_tips_from_ply(gt_path)
        all_gt_tips.append(gt_tip_coords)

        # Initialize tracking for tips appearing at this timepoint
        if len(gt_tip_coords) > len(tracked_tips):
            num_new_tips = len(gt_tip_coords) - len(tracked_tips)
            for i in range(num_new_tips):
                new_tip_id = len(tracked_tips)
                new_tip_coord = gt_tip_coords[new_tip_id]
                tracked_tips[new_tip_id] = (t_idx, new_tip_coord, normalized_times[t_idx])
                logging.info(f"Initialized tip {new_tip_id} appearing at timepoint {t_idx}")

    # Now track tips through time
    epe_values_all = []
    epe_values_train = [] if point_splits else None
    epe_values_test = [] if point_splits else None

    for t_idx, gt_path in enumerate(gt_paths):
        target_time = normalized_times[t_idx]
        gt_tip_coords = all_gt_tips[t_idx]

        if len(gt_tip_coords) == 0:
            logging.info(f"Timepoint {t_idx}: No GT tips found, skipping")
            epe_values_all.append(None)
            if point_splits:
                epe_values_train.append(None)
                epe_values_test.append(None)
            continue

        # Warp all tracked tips to target time
        warped_tip_coords_list = []

        for tip_id in sorted(tracked_tips.keys()):
            first_appear_idx, first_appear_coord, first_appear_time = tracked_tips[tip_id]

            if t_idx < first_appear_idx:
                # Tip hasn't appeared yet
                continue
            elif t_idx == first_appear_idx:
                # Tip just appeared, no warping needed
                warped_tip_coords_list.append(first_appear_coord)
            else:
                # Warp from first appearance to target time
                tip_coord_normalized = (first_appear_coord - mean.numpy()) * scale.numpy()
                tip_tensor = torch.from_numpy(tip_coord_normalized.reshape(1, 3)).float().cuda()

                with torch.no_grad():
                    warped_normalized = decoder.warper(
                        tip_tensor,
                        first_appear_time.cuda(),
                        target_time.cuda()
                    )

                # Denormalize
                warped_coord = (warped_normalized[0].cpu().numpy() / scale.numpy()) + mean.numpy()
                warped_coord = warped_coord.flatten()
                warped_tip_coords_list.append(warped_coord)

        warped_tip_coords = np.array(warped_tip_coords_list)

        # Compute EPE
        if len(warped_tip_coords) == len(gt_tip_coords):
            epe_per_tip = np.linalg.norm(warped_tip_coords - gt_tip_coords, axis=1)
            mean_epe = epe_per_tip.mean()
            epe_values_all.append(mean_epe)

            logging.info(
                f"Timepoint {t_idx}: Tracked {len(warped_tip_coords)} tips, Mean EPE = {mean_epe:.6f}"
            )

            # Handle splits if provided
            if point_splits:
                # Load GT points to check which split tips belong to
                gt_points, _, _ = pcu.load_mesh_vfn(gt_path, dtype=np.float32)
                gt_points, _, _ = pcu.deduplicate_point_cloud(gt_points, 1e-15, return_index=True)

                timepoint_data = point_splits["timepoints"][t_idx]
                train_indices = np.array(timepoint_data["train_indices"], dtype=np.int64)
                test_indices = np.array(timepoint_data["test_indices"], dtype=np.int64)

                # Load tip indices from GT
                gt_tip_indices_in_full_cloud, _ = load_leaf_tips_from_ply(gt_path)

                # Separate tips into train and test
                train_tip_mask = np.isin(gt_tip_indices_in_full_cloud, train_indices)
                test_tip_mask = np.isin(gt_tip_indices_in_full_cloud, test_indices)

                # Compute EPE for train tips
                if train_tip_mask.sum() > 0:
                    train_epe = epe_per_tip[train_tip_mask].mean()
                    epe_values_train.append(train_epe)
                else:
                    epe_values_train.append(None)

                # Compute EPE for test tips
                if test_tip_mask.sum() > 0:
                    test_epe = epe_per_tip[test_tip_mask].mean()
                    epe_values_test.append(test_epe)
                else:
                    epe_values_test.append(None)
        else:
            logging.warning(
                f"Timepoint {t_idx}: Tip count mismatch "
                f"(warped: {len(warped_tip_coords)}, GT: {len(gt_tip_coords)})"
            )
            epe_values_all.append(None)
            if point_splits:
                epe_values_train.append(None)
                epe_values_test.append(None)

    result = {"all": epe_values_all}
    if point_splits:
        result["train"] = epe_values_train
        result["test"] = epe_values_test

    return result


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
        pred_normals_train_list = [] if args.compute_normal_consistency else None
        gt_normals_train_list = [] if args.compute_normal_consistency else None
        pred_normals_test_list = [] if args.compute_normal_consistency else None
        gt_normals_test_list = [] if args.compute_normal_consistency else None
        pred_tips_train_list = []
        gt_tips_train_list = []
        pred_tips_test_list = []
        gt_tips_test_list = []
    else:
        pred_points_list = []
        gt_points_list = []
        pred_normals_list = [] if args.compute_normal_consistency else None
        gt_normals_list = [] if args.compute_normal_consistency else None

    # Always initialize tips lists (used regardless of splits)
    pred_tips_list = []
    gt_tips_list = []

    # Load all meshes and compute per-timestep metrics
    for i, (pred_file, gt_path) in enumerate(zip(pred_files, gt_paths)):
        pred_path = os.path.join(pred_mesh_dir, pred_file)

        logging.info(f"Processing timestep {i+1}/{len(pred_files)}: {pred_file}")

        # Load predicted mesh as points (and normals if needed)
        try:
            if args.compute_normal_consistency:
                pred_points, pred_normals = load_mesh_as_points(pred_path, args.num_samples, return_normals=True)
            else:
                pred_points = load_mesh_as_points(pred_path, args.num_samples, return_normals=False)
                pred_normals = None
        except Exception as e:
            logging.error(f"Error loading predicted mesh {pred_file}: {e}")
            continue

        # Load ground truth point cloud (and normals if needed)
        try:
            if args.compute_normal_consistency:
                gt_points, _, gt_normals = pcu.load_mesh_vfn(gt_path, dtype=np.float32)
                # Deduplicate points and normals together
                gt_points, unique_indices, _ = pcu.deduplicate_point_cloud(
                    gt_points, 1e-15, return_index=True
                )
                gt_normals = gt_normals[unique_indices]
            else:
                gt_points, _, _ = pcu.load_mesh_vfn(gt_path, dtype=np.float32)
                # Deduplicate
                gt_points, _, _ = pcu.deduplicate_point_cloud(
                    gt_points, 1e-15, return_index=True
                )
                gt_normals = None
        except Exception as e:
            logging.error(f"Error loading ground truth {gt_path}: {e}")
            continue

        # Apply point splits if provided
        if point_splits:
            timepoint_data = point_splits["timepoints"][i]
            train_indices = np.array(timepoint_data["train_indices"], dtype=np.int64)
            test_indices = np.array(timepoint_data["test_indices"], dtype=np.int64)

            # DEBUG: Verify indices are loaded correctly
            if i == 0:  # Only print for first timestep
                logging.info(f"DEBUG: First 10 train indices from split file: {train_indices[:10]}")
                logging.info(f"DEBUG: First 10 test indices from split file: {test_indices[:10]}")
                logging.info(f"DEBUG: Total GT points: {len(gt_points)}")
                logging.info(f"DEBUG: n_train from split file: {timepoint_data.get('n_train', 'N/A')}")
                logging.info(f"DEBUG: n_test from split file: {timepoint_data.get('n_test', 'N/A')}")

            # Apply splits to GT points
            gt_points_train = apply_point_split(gt_points, train_indices)
            gt_points_test = apply_point_split(gt_points, test_indices)

            # For predicted points, we need to use the same split indices
            # Note: pred_points may have different number of points, so we need to be careful
            pred_points_train = apply_point_split(pred_points, train_indices)
            pred_points_test = apply_point_split(pred_points, test_indices)

            # Convert to tensors
            pred_points_train_tensor = torch.tensor(
                pred_points_train, dtype=torch.float32
            ).cuda()
            gt_points_train_tensor = torch.tensor(
                gt_points_train, dtype=torch.float32
            ).cuda()
            pred_points_test_tensor = torch.tensor(
                pred_points_test, dtype=torch.float32
            ).cuda()
            gt_points_test_tensor = torch.tensor(
                gt_points_test, dtype=torch.float32
            ).cuda()

            pred_points_train_list.append(pred_points_train_tensor)
            gt_points_train_list.append(gt_points_train_tensor)
            pred_points_test_list.append(pred_points_test_tensor)
            gt_points_test_list.append(gt_points_test_tensor)

            # Handle normals if needed
            if args.compute_normal_consistency:
                gt_normals_train = apply_point_split(gt_normals, train_indices)
                gt_normals_test = apply_point_split(gt_normals, test_indices)
                pred_normals_train = apply_point_split(pred_normals, train_indices)
                pred_normals_test = apply_point_split(pred_normals, test_indices)

                pred_normals_train_tensor = torch.tensor(pred_normals_train, dtype=torch.float32).cuda()
                gt_normals_train_tensor = torch.tensor(gt_normals_train, dtype=torch.float32).cuda()
                pred_normals_test_tensor = torch.tensor(pred_normals_test, dtype=torch.float32).cuda()
                gt_normals_test_tensor = torch.tensor(gt_normals_test, dtype=torch.float32).cuda()

                pred_normals_train_list.append(pred_normals_train_tensor)
                gt_normals_train_list.append(gt_normals_train_tensor)
                pred_normals_test_list.append(pred_normals_test_tensor)
                gt_normals_test_list.append(gt_normals_test_tensor)

            logging.info(
                f"  Train points: pred={len(pred_points_train)}, gt={len(gt_points_train)}"
            )
            logging.info(
                f"  Test points: pred={len(pred_points_test)}, gt={len(gt_points_test)}"
            )

            # Load and split leaf tips if requested
            if args.compute_tip_epe:
                gt_tip_indices_in_full_cloud, gt_tip_coords = load_leaf_tips_from_ply(gt_path)

                if len(gt_tip_indices_in_full_cloud) > 0:

                    # Separate tips into train and test based on which split they fall into
                    train_tip_mask = np.isin(
                        gt_tip_indices_in_full_cloud, train_indices
                    )
                    test_tip_mask = np.isin(gt_tip_indices_in_full_cloud, test_indices)

                    # DEBUG: Check which split the tips fall into
                    n_tips_in_train = train_tip_mask.sum()
                    n_tips_in_test = test_tip_mask.sum()
                    logging.info(f"  Timestep {i}: Found {len(gt_tip_coords)} tips total")
                    logging.info(f"    Tips in train split: {n_tips_in_train}")
                    logging.info(f"    Tips in test split: {n_tips_in_test}")
                    if n_tips_in_train > 0:
                        tip_indices_in_train = gt_tip_indices_in_full_cloud[train_tip_mask]
                        logging.info(f"    Train tip indices in full cloud: {tip_indices_in_train}")
                    if n_tips_in_test > 0:
                        tip_indices_in_test = gt_tip_indices_in_full_cloud[test_tip_mask]
                        logging.info(f"    Test tip indices in full cloud: {tip_indices_in_test}")

                    train_tip_indices_in_split = []
                    test_tip_indices_in_split = []

                    # For train tips: find their positions in the train-split array
                    for tip_idx in gt_tip_indices_in_full_cloud[train_tip_mask]:
                        pos_in_train = np.where(train_indices == tip_idx)[0]
                        if len(pos_in_train) > 0:
                            train_tip_indices_in_split.append(pos_in_train[0])

                    # For test tips: find their positions in the test-split array
                    for tip_idx in gt_tip_indices_in_full_cloud[test_tip_mask]:
                        pos_in_test = np.where(test_indices == tip_idx)[0]
                        if len(pos_in_test) > 0:
                            test_tip_indices_in_split.append(pos_in_test[0])

                    # Extract GT tips from split point clouds
                    if len(train_tip_indices_in_split) > 0:
                        gt_tips_train = gt_points_train[train_tip_indices_in_split]
                        gt_tips_train_tensor = torch.tensor(
                            gt_tips_train, dtype=torch.float32
                        ).cuda()

                        # Find corresponding predicted tips (nearest points)
                        pred_tips_train_tensor = torch.zeros_like(gt_tips_train_tensor)
                        for j, gt_tip in enumerate(gt_tips_train_tensor):
                            distances = torch.norm(
                                pred_points_train_tensor - gt_tip.unsqueeze(0), dim=1
                            )
                            nearest_idx = torch.argmin(distances)
                            pred_tips_train_tensor[j] = pred_points_train_tensor[
                                nearest_idx
                            ]

                        pred_tips_train_list.append(pred_tips_train_tensor)
                        gt_tips_train_list.append(gt_tips_train_tensor)
                    else:
                        pred_tips_train_list.append(None)
                        gt_tips_train_list.append(None)

                    if len(test_tip_indices_in_split) > 0:
                        gt_tips_test = gt_points_test[test_tip_indices_in_split]
                        gt_tips_test_tensor = torch.tensor(
                            gt_tips_test, dtype=torch.float32
                        ).cuda()

                        # Find corresponding predicted tips (nearest points)
                        pred_tips_test_tensor = torch.zeros_like(gt_tips_test_tensor)
                        for j, gt_tip in enumerate(gt_tips_test_tensor):
                            distances = torch.norm(
                                pred_points_test_tensor - gt_tip.unsqueeze(0), dim=1
                            )
                            nearest_idx = torch.argmin(distances)
                            pred_tips_test_tensor[j] = pred_points_test_tensor[
                                nearest_idx
                            ]

                        pred_tips_test_list.append(pred_tips_test_tensor)
                        gt_tips_test_list.append(gt_tips_test_tensor)
                    else:
                        pred_tips_test_list.append(None)
                        gt_tips_test_list.append(None)
                else:
                    # No tips in this timestep
                    pred_tips_train_list.append(None)
                    gt_tips_train_list.append(None)
                    pred_tips_test_list.append(None)
                    gt_tips_test_list.append(None)
        else:
            # No splits - use all points
            pred_points_tensor = torch.tensor(pred_points, dtype=torch.float32).cuda()
            gt_points_tensor = torch.tensor(gt_points, dtype=torch.float32).cuda()

            pred_points_list.append(pred_points_tensor)
            gt_points_list.append(gt_points_tensor)

            # Handle normals if needed
            if args.compute_normal_consistency:
                pred_normals_tensor = torch.tensor(pred_normals, dtype=torch.float32).cuda()
                gt_normals_tensor = torch.tensor(gt_normals, dtype=torch.float32).cuda()
                pred_normals_list.append(pred_normals_tensor)
                gt_normals_list.append(gt_normals_tensor)

            # Load leaf tips from ground truth PLY file if requested (no-split case)
            if args.compute_tip_epe:
                # Load GT leaf tips from PLY file
                _, gt_tip_coords = load_leaf_tips_from_ply(gt_path)

                if len(gt_tip_coords) > 0:
                    # Convert to tensor
                    gt_tips_tensor = torch.tensor(
                        gt_tip_coords, dtype=torch.float32
                    ).cuda()

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

        # Train metrics - Chamfer Distance using PyTorch3D
        cd_train_values = []
        for pred_pts, gt_pts in zip(pred_points_train_list, gt_points_train_list):
            cd = compute_pytorch3d_chamfer_distance(pred_pts, gt_pts)
            cd_train_values.append(cd)

        cd_train_tensor = torch.tensor(cd_train_values)
        metrics_json["train"] = {
            "num_timesteps": len(pred_points_train_list),
            "mean_chamfer_distance": float(cd_train_tensor.mean().item()),
            "std_chamfer_distance": float(cd_train_tensor.std().item()),
            "chamfer_distances_per_timestep": [float(cd) for cd in cd_train_values],
        }

        # Train metrics - Normal Consistency
        if args.compute_normal_consistency:
            nc_train_values = []
            for pred_pts, pred_norms, gt_pts, gt_norms in zip(
                pred_points_train_list, pred_normals_train_list,
                gt_points_train_list, gt_normals_train_list
            ):
                nc = compute_normal_consistency(pred_pts, pred_norms, gt_pts, gt_norms)
                nc_train_values.append(nc)

            nc_train_tensor = torch.tensor(nc_train_values)
            metrics_json["train"]["mean_normal_consistency"] = float(nc_train_tensor.mean().item())
            metrics_json["train"]["std_normal_consistency"] = float(nc_train_tensor.std().item())
            metrics_json["train"]["normal_consistency_per_timestep"] = [float(nc) for nc in nc_train_values]

        # Test metrics - Chamfer Distance using PyTorch3D
        cd_test_values = []
        for pred_pts, gt_pts in zip(pred_points_test_list, gt_points_test_list):
            cd = compute_pytorch3d_chamfer_distance(pred_pts, gt_pts)
            cd_test_values.append(cd)

        cd_test_tensor = torch.tensor(cd_test_values)
        metrics_json["test"] = {
            "num_timesteps": len(pred_points_test_list),
            "mean_chamfer_distance": float(cd_test_tensor.mean().item()),
            "std_chamfer_distance": float(cd_test_tensor.std().item()),
            "chamfer_distances_per_timestep": [float(cd) for cd in cd_test_values],
        }

        # Test metrics - Normal Consistency
        if args.compute_normal_consistency:
            nc_test_values = []
            for pred_pts, pred_norms, gt_pts, gt_norms in zip(
                pred_points_test_list, pred_normals_test_list,
                gt_points_test_list, gt_normals_test_list
            ):
                nc = compute_normal_consistency(pred_pts, pred_norms, gt_pts, gt_norms)
                nc_test_values.append(nc)

            nc_test_tensor = torch.tensor(nc_test_values)
            metrics_json["test"]["mean_normal_consistency"] = float(nc_test_tensor.mean().item())
            metrics_json["test"]["std_normal_consistency"] = float(nc_test_tensor.std().item())
            metrics_json["test"]["normal_consistency_per_timestep"] = [float(nc) for nc in nc_test_values]

        # Compute EPE for leaf tips using warper-based tracking
        if args.compute_tip_epe:
            try:
                # Load specs and dataset for warper
                from datasets import data_pheno4d

                specs_path = args.specifications
                if not os.path.isabs(specs_path):
                    specs_path = os.path.abspath(specs_path)

                specs = json.load(open(specs_path))

                # Load dataset
                split_file = args.point_split_file
                if args.dataset_name == "pheno4d" and split_file is None:
                    sequence_name = args.object_name.split("/")[-1]
                    split_file = f"configs/point_splits/{sequence_name}_split.json"

                sdf_dataset = data_pheno4d.DeformSamples(
                    args.object_name,
                    start_idx=0,
                    gap=None,
                    train_step=1,
                    use_dense=False,
                    split_file=split_file,
                    use_train_points=True,
                )

                # Load decoder model
                decoder, scale, mean, normalized_times = load_decoder_model(args, specs, sdf_dataset, epoch_num)

                # Track tips with warper
                tip_epe_results = track_tips_with_warper(
                    decoder, gt_paths, sdf_dataset, scale, mean, normalized_times, point_splits
                )

                # Add results to metrics_json
                # Train tips
                train_epe_values = [v for v in tip_epe_results["train"] if v is not None]
                if len(train_epe_values) > 0:
                    train_epe_tensor = torch.tensor(train_epe_values)
                    metrics_json["train"]["num_timesteps_with_tips"] = len(train_epe_values)
                    metrics_json["train"]["mean_tip_epe"] = float(train_epe_tensor.mean().item())
                    metrics_json["train"]["std_tip_epe"] = float(train_epe_tensor.std().item())
                    metrics_json["train"]["tip_epe_per_timestep"] = [float(v) for v in tip_epe_results["train"]]
                    logging.info(
                        f"Train Set - Leaf Tip EPE (warper): {metrics_json['train']['mean_tip_epe']:.6f}"
                    )
                else:
                    metrics_json["train"]["num_timesteps_with_tips"] = 0
                    metrics_json["train"]["mean_tip_epe"] = None
                    metrics_json["train"]["std_tip_epe"] = None
                    metrics_json["train"]["tip_epe_per_timestep"] = []
                    logging.info("Train Set - No leaf tips found (expected if tips are in test set)")

                # Test tips
                test_epe_values = [v for v in tip_epe_results["test"] if v is not None]
                if len(test_epe_values) > 0:
                    test_epe_tensor = torch.tensor(test_epe_values)
                    metrics_json["test"]["num_timesteps_with_tips"] = len(test_epe_values)
                    metrics_json["test"]["mean_tip_epe"] = float(test_epe_tensor.mean().item())
                    metrics_json["test"]["std_tip_epe"] = float(test_epe_tensor.std().item())
                    metrics_json["test"]["tip_epe_per_timestep"] = [float(v) for v in tip_epe_results["test"]]
                    logging.info(
                        f"Test Set - Leaf Tip EPE (warper): {metrics_json['test']['mean_tip_epe']:.6f}"
                    )
                else:
                    metrics_json["test"]["num_timesteps_with_tips"] = 0
                    metrics_json["test"]["mean_tip_epe"] = None
                    metrics_json["test"]["std_tip_epe"] = None
                    metrics_json["test"]["tip_epe_per_timestep"] = []
                    logging.info("Test Set - No leaf tips found")

            except Exception as e:
                logging.error(f"Error computing warper-based tip EPE: {e}")
                logging.warning("Falling back to skipping tip EPE computation")
                metrics_json["train"]["num_timesteps_with_tips"] = 0
                metrics_json["train"]["mean_tip_epe"] = None
                metrics_json["train"]["std_tip_epe"] = None
                metrics_json["train"]["tip_epe_per_timestep"] = []
                metrics_json["test"]["num_timesteps_with_tips"] = 0
                metrics_json["test"]["mean_tip_epe"] = None
                metrics_json["test"]["std_tip_epe"] = None
                metrics_json["test"]["tip_epe_per_timestep"] = []

        # Log results
        logging.info(
            f"Train Set - Mean CD: {metrics_json['train']['mean_chamfer_distance']:.6f}"
        )
        logging.info(
            f"Train Set - Std CD: {metrics_json['train']['std_chamfer_distance']:.6f}"
        )
        logging.info(
            f"Test Set - Mean CD: {metrics_json['test']['mean_chamfer_distance']:.6f}"
        )
        logging.info(
            f"Test Set - Std CD: {metrics_json['test']['std_chamfer_distance']:.6f}"
        )

    else:
        # No splits - compute metrics on all points (backward compatible)
        logging.info("Computing metrics on full point clouds...")

        # Chamfer Distance using PyTorch3D
        cd_values = []
        for pred_pts, gt_pts in zip(pred_points_list, gt_points_list):
            cd = compute_pytorch3d_chamfer_distance(pred_pts, gt_pts)
            cd_values.append(cd)

        cd_tensor = torch.tensor(cd_values)
        metrics_json["num_timesteps"] = len(pred_points_list)
        metrics_json["mean_chamfer_distance"] = float(cd_tensor.mean().item())
        metrics_json["std_chamfer_distance"] = float(cd_tensor.std().item())
        metrics_json["chamfer_distances_per_timestep"] = [float(cd) for cd in cd_values]

        # Normal Consistency
        if args.compute_normal_consistency:
            nc_values = []
            for pred_pts, pred_norms, gt_pts, gt_norms in zip(
                pred_points_list, pred_normals_list, gt_points_list, gt_normals_list
            ):
                nc = compute_normal_consistency(pred_pts, pred_norms, gt_pts, gt_norms)
                nc_values.append(nc)

            nc_tensor = torch.tensor(nc_values)
            metrics_json["mean_normal_consistency"] = float(nc_tensor.mean().item())
            metrics_json["std_normal_consistency"] = float(nc_tensor.std().item())
            metrics_json["normal_consistency_per_timestep"] = [float(nc) for nc in nc_values]

        # Tip EPE using warper-based tracking
        if args.compute_tip_epe:
            try:
                # Load specs and dataset for warper
                from datasets import data_pheno4d

                specs_path = args.specifications
                if not os.path.isabs(specs_path):
                    specs_path = os.path.abspath(specs_path)

                specs = json.load(open(specs_path))

                # Load dataset
                split_file = args.point_split_file
                if args.dataset_name == "pheno4d" and split_file is None:
                    sequence_name = args.object_name.split("/")[-1]
                    split_file = f"configs/point_splits/{sequence_name}_split.json"
                    if not os.path.exists(split_file):
                        split_file = None

                sdf_dataset = data_pheno4d.DeformSamples(
                    args.object_name,
                    start_idx=0,
                    gap=None,
                    train_step=1,
                    use_dense=False,
                    split_file=split_file,
                    use_train_points=True,
                )

                # Load decoder model
                decoder, scale, mean, normalized_times = load_decoder_model(args, specs, sdf_dataset, epoch_num)

                # Track tips with warper (no splits)
                tip_epe_results = track_tips_with_warper(
                    decoder, gt_paths, sdf_dataset, scale, mean, normalized_times, point_splits=None
                )

                # Add results to metrics_json
                epe_values = [v for v in tip_epe_results["all"] if v is not None]
                if len(epe_values) > 0:
                    epe_tensor = torch.tensor(epe_values)
                    metrics_json["num_timesteps_with_tips"] = len(epe_values)
                    metrics_json["mean_tip_epe"] = float(epe_tensor.mean().item())
                    metrics_json["std_tip_epe"] = float(epe_tensor.std().item())
                    metrics_json["tip_epe_per_timestep"] = [float(v) for v in tip_epe_results["all"]]
                    logging.info(f"Leaf Tip EPE (warper): {metrics_json['mean_tip_epe']:.6f}")
                else:
                    metrics_json["num_timesteps_with_tips"] = 0
                    metrics_json["mean_tip_epe"] = None
                    metrics_json["std_tip_epe"] = None
                    metrics_json["tip_epe_per_timestep"] = []
                    logging.info("No leaf tips found in any timestep")

            except Exception as e:
                logging.error(f"Error computing warper-based tip EPE: {e}")
                logging.warning("Skipping tip EPE computation")
                metrics_json["num_timesteps_with_tips"] = 0
                metrics_json["mean_tip_epe"] = None
                metrics_json["std_tip_epe"] = None
                metrics_json["tip_epe_per_timestep"] = []

        # Log results
        logging.info(
            f"Mean Chamfer Distance: {metrics_json['mean_chamfer_distance']:.6f}"
        )
        logging.info(
            f"Std Chamfer Distance: {metrics_json['std_chamfer_distance']:.6f}"
        )
        if args.compute_normal_consistency:
            logging.info(f"Mean Normal Consistency: {metrics_json['mean_normal_consistency']:.6f}")
            logging.info(f"Std Normal Consistency: {metrics_json['std_normal_consistency']:.6f}")
        if args.compute_tip_epe and "mean_tip_epe" in metrics_json and metrics_json["mean_tip_epe"] is not None:
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
