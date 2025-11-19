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
from pathlib import Path

import numpy as np
import open3d as o3d
import point_cloud_utils as pcu
import torch
import trimesh
from pytorch3d.loss import chamfer_distance

from metrics.metrics import PlantMetrics
from utils import (
    add_common_args,
    configure_logging,
    load_latent_vectors,
    load_model_parameters,
)


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


def load_mesh_as_points(
    mesh_path: str, num_samples: int = 10000, return_normals: bool = False
):
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
            logging.info(f"No is_leaf_tip attribute in {Path(ply_path).name}")
            return np.array([]), np.array([])
    except Exception as e:
        logging.warning(f"Exception loading tips from {ply_path}: {e}")
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
        interp_stride = None

        for i, part in enumerate(parts):
            if part == "start" and i + 2 < len(parts) and parts[i + 1] == "id":
                start_idx = int(parts[i + 2])
            elif part == "gap" and i + 1 < len(parts):
                gap = int(parts[i + 1])
            elif part == "stride" and i + 1 < len(parts):
                # Check if this is interp_stride or regular stride
                if i > 0 and parts[i - 1] == "interp":
                    interp_stride = int(parts[i + 1])
                else:
                    stride = int(parts[i + 1])

        # Apply same filtering as dataset
        end_idx = min(start_idx + gap, len(gt_files))

        if interp_stride is not None:
            # Interpolation mode: load all files, then filter to TEST timesteps
            all_files = gt_files[start_idx:end_idx]
            # Only keep TEST timesteps (positions 1, 3, 5, ...)
            test_indices = [i for i in range(len(all_files)) if i % interp_stride != 0]
            test_gt_files = [all_files[i] for i in test_indices]
            logging.info(f"Interpolation mode detected (interp_stride={interp_stride})")
            logging.info(f"  Total timesteps: {len(all_files)}")
            logging.info(
                f"  Evaluating on {len(test_gt_files)} TEST timesteps: {test_indices}"
            )
            gt_files = test_gt_files
            original_indices = test_indices
        else:
            # Normal mode: apply stride filtering
            gt_files = gt_files[start_idx:end_idx:stride]
            original_indices = list(range(len(gt_files)))

        gt_paths = [os.path.join(pc_path, f) for f in gt_files]
        return gt_paths, gt_files, original_indices
    else:
        raise NotImplementedError(
            f"Ground truth loading not implemented for {args.dataset_name}"
        )


def compute_pytorch3d_chamfer_distance(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    pred_normals: torch.Tensor = None,
    gt_normals: torch.Tensor = None,
) -> tuple:
    """
    Compute Chamfer Distance and optionally Normal Consistency using PyTorch3D.

    Args:
        pred_points: Predicted point cloud of shape (N, 3)
        gt_points: Ground truth point cloud of shape (M, 3)
        pred_normals: Optional predicted normals of shape (N, 3)
        gt_normals: Optional GT normals of shape (M, 3)

    Returns:
        Tuple of (chamfer_distance, normal_consistency)
        - chamfer_distance: float, sum of bidirectional distances
        - normal_consistency: float or None, 1 - cosine_distance (1=aligned, 0=orthogonal)
    """
    # PyTorch3D expects batch dimension
    pred_batch = pred_points.unsqueeze(0)  # (1, N, 3)
    gt_batch = gt_points.unsqueeze(0)  # (1, M, 3)

    # Prepare normals if provided
    pred_normals_batch = pred_normals.unsqueeze(0) if pred_normals is not None else None
    gt_normals_batch = gt_normals.unsqueeze(0) if gt_normals is not None else None

    # chamfer_distance returns (loss_points, loss_normals) where:
    # loss_points: sum of bidirectional point distances
    # loss_normals: cosine distance of normals (if normals provided), else None
    cd_loss, normal_loss = chamfer_distance(
        pred_batch, gt_batch, x_normals=pred_normals_batch, y_normals=gt_normals_batch
    )

    # Convert cosine distance to consistency: 1 - cosine_distance
    # (cosine distance ranges from 0=aligned to 2=opposite, so 1-cosine_dist gives consistency)
    normal_consistency = None
    if normal_loss is not None:
        normal_consistency = (1.0 - normal_loss).item()

    return cd_loss.item(), normal_consistency


def scene_flow_metrics(pred, labels, strict=0.025, relax=0.05):
    """
    Compute scene flow metrics: EPE3D, AccS (Strict Accuracy), AccR (Relaxed Accuracy).

    Args:
        pred: Predicted flow vectors of shape (N, 3)
        labels: Ground truth flow vectors of shape (N, 3)
        strict: Strict threshold (default 0.025 = 2.5cm)
        relax: Relaxed threshold (default 0.05 = 5cm)

    Returns:
        Tuple of (EPE3D, AccS, AccR) where AccS and AccR are in percentage [0-100]
    """
    l2_norm = torch.sqrt(
        torch.sum((pred - labels) ** 2, 1)
    ).cpu()  # Absolute distance error.
    labels_norm = torch.sqrt(torch.sum(labels * labels, 1)).cpu()
    relative_err = l2_norm / (labels_norm + 1e-20)

    EPE3D = torch.mean(l2_norm).item()  # Mean absolute distance error

    # NOTE: AccS (Strict Accuracy)
    error_lt_5 = torch.BoolTensor((l2_norm < strict))
    relative_err_lt_5 = torch.BoolTensor((relative_err < strict))
    AccS = torch.mean((error_lt_5 | relative_err_lt_5).float()).item()

    # NOTE: AccR (Relaxed Accuracy)
    error_lt_10 = torch.BoolTensor((l2_norm < relax))
    relative_err_lt_10 = torch.BoolTensor((relative_err < relax))
    AccR = torch.mean((error_lt_10 | relative_err_lt_10).float()).item()

    return EPE3D, AccS * 100, AccR * 100


def compute_storage_size(experiment_directory, epoch):
    """
    Compute total storage size of model checkpoints for a given epoch.

    Args:
        experiment_directory: Path to experiment directory
        epoch: Epoch number (string)

    Returns:
        Total size in MB
    """
    checkpoint_dir = os.path.join(experiment_directory, "checkpoints")
    total_size_bytes = 0

    # Files are in subdirectories: checkpoints/{AdjustLatentCodes,LatentCodes,ModelParameters}/{epoch}.pth
    checkpoint_subdirs = ["AdjustLatentCodes", "LatentCodes", "ModelParameters"]

    for subdir in checkpoint_subdirs:
        filepath = os.path.join(checkpoint_dir, subdir, f"{epoch}.pth")
        if os.path.exists(filepath):
            total_size_bytes += os.path.getsize(filepath)

    # Convert to MB
    return total_size_bytes / (1024 * 1024)


def get_runtime_from_wandb(experiment_directory):
    """
    Extract runtime from wandb summary file.

    Args:
        experiment_directory: Path to experiment directory

    Returns:
        Runtime in minutes, or None if not found
    """
    wandb_summary_path = os.path.join(
        experiment_directory, "wandb", "latest-run", "files", "wandb-summary.json"
    )

    if not os.path.exists(wandb_summary_path):
        logging.warning(f"wandb summary not found: {wandb_summary_path}")
        return None

    try:
        with open(wandb_summary_path, "r") as f:
            summary = json.load(f)

        # Extract runtime from _wandb.runtime (in seconds)
        runtime_seconds = summary.get("_wandb", {}).get("runtime")
        if runtime_seconds is not None:
            return runtime_seconds / 60.0  # Convert to minutes
        else:
            logging.warning("Runtime not found in wandb summary")
            return None
    except Exception as e:
        logging.warning(f"Error reading wandb summary: {e}")
        return None


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
    # Import the correct Decoder based on method
    method = args.methods
    if method in ["ode_solve", "ode_solve_woAdaptor"]:
        from networks.multi_velocity_debug import Decoder, adjust_net
    else:
        # NVFI, 4DSDF, OF4D use other_volocities
        from networks.other_volocities import Decoder

        adjust_net = None  # These methods don't use adjust network

    logging.info("Loading trained decoder model for tip tracking...")

    latent_size = specs["CodeLength"]

    # Create adjustment network (only for ode_solve)
    if method in ["ode_solve", "ode_solve_woAdaptor"]:
        ajust = adjust_net().cuda()
    else:
        ajust = None

    # Create decoder with the actual method
    method = args.methods  # Use the method specified in args
    decoder = Decoder(
        method,
        latent_size,
        specs["NetworkSpecs"]["warper_kargs"],
        specs["NetworkSpecs"]["decoder_debug_kargs"],
    )
    decoder = decoder.cuda()

    # Create embeddings and equip them BEFORE loading (so model structure matches checkpoint)
    experiment_directory = get_experiment_directory(args)
    epoch = str(epoch_num)  # Convert to string for load_model_parameters

    # Create embeddings for methods that use them
    if method in ["ode_solve", "ode_solve_woAdaptor", "OF4D"]:
        # These methods use 2 embeddings (start and end)
        start_end_vecs = torch.nn.Embedding(2, 128, max_norm=1.0).cuda()
        decoder.warper.equip_embedding(start_end_vecs)
    elif method == "TPSR":
        # TPSR uses 1 embedding (shape code)
        start_end_vecs = torch.nn.Embedding(1, 128, max_norm=1.0).cuda()
        decoder.warper.equip_embedding(start_end_vecs)
    else:
        start_end_vecs = None

    # Load model
    if method in ["ode_solve", "ode_solve_woAdaptor"]:
        # ode_solve uses adjust network
        composed_model = torch.nn.ModuleList().cuda()
        composed_model.append(ajust)
        composed_model.append(decoder)
        load_model_parameters(experiment_directory, epoch, composed_model)
        ajust = composed_model[0]
        decoder = composed_model[1]

        # Load latent vectors for ode_solve
        load_latent_vectors(experiment_directory, epoch, start_end_vecs)
    else:
        # Other methods (NVFI, 4DSDF, OF4D, TPSR) only have decoder
        load_model_parameters(experiment_directory, epoch, decoder)

        # Load latent vectors for methods that have separate embeddings
        if method in ["OF4D", "TPSR"]:
            load_latent_vectors(experiment_directory, epoch, start_end_vecs)

    decoder.eval()
    if ajust is not None:
        ajust.eval()

    # Set time range (for methods that use time splitting)
    if method in ["ode_solve", "ode_solve_woAdaptor", "TPSR"]:
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
    point_splits=None,
    original_indices=None,
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
        original_indices: Optional list mapping filtered indices to original indices (for interpolation mode)

    Returns:
        Dictionary with tracked tips and EPE values per split
    """
    logging.info("Tracking leaf tips with warper...")

    # If original_indices not provided, assume sequential (normal mode)
    if original_indices is None:
        original_indices = list(range(len(gt_paths)))

    # Dictionary to track tips: {tip_id: (first_appear_idx, first_appear_coords, first_appear_time)}
    tracked_tips = {}

    # Load all GT tips first to identify unique tips
    all_gt_tips = []
    logging.info(f"Loading tips from {len(gt_paths)} timesteps for tip tracking...")
    for t_idx, gt_path in enumerate(gt_paths):
        _, gt_tip_coords = load_leaf_tips_from_ply(gt_path)
        all_gt_tips.append(gt_tip_coords)

        # Initialize tracking for tips appearing at this timepoint
        if len(gt_tip_coords) > len(tracked_tips):
            num_new_tips = len(gt_tip_coords) - len(tracked_tips)
            for i in range(num_new_tips):
                new_tip_id = len(tracked_tips)
                new_tip_coord = gt_tip_coords[new_tip_id]
                tracked_tips[new_tip_id] = (
                    t_idx,
                    new_tip_coord,
                    normalized_times[t_idx],
                )

    # Now track tips through time
    epe_values_all = []
    epe_values_train = [] if point_splits else None
    epe_values_test = [] if point_splits else None

    # Also store warped and GT coordinates for AccS/AccR computation
    warped_coords_all = []
    gt_coords_all = []
    warped_coords_test = [] if point_splits else None
    gt_coords_test = [] if point_splits else None

    for t_idx, gt_path in enumerate(gt_paths):
        target_time = normalized_times[t_idx]
        gt_tip_coords = all_gt_tips[t_idx]

        if len(gt_tip_coords) == 0:
            logging.info(f"Timepoint {t_idx}: No GT tips found, skipping")
            epe_values_all.append(None)
            warped_coords_all.append(None)
            gt_coords_all.append(None)
            if point_splits:
                epe_values_train.append(None)
                epe_values_test.append(None)
                warped_coords_test.append(None)
                gt_coords_test.append(None)
            continue

        # Warp all tracked tips to target time
        warped_tip_coords_list = []

        for tip_id in sorted(tracked_tips.keys()):
            first_appear_idx, first_appear_coord, first_appear_time = tracked_tips[
                tip_id
            ]

            if t_idx < first_appear_idx:
                # Tip hasn't appeared yet
                continue
            elif t_idx == first_appear_idx:
                # Tip just appeared, no warping needed
                warped_tip_coords_list.append(first_appear_coord)
            else:
                # Warp from first appearance to target time
                tip_coord_normalized = (
                    first_appear_coord - mean.numpy()
                ) * scale.numpy()
                tip_tensor = (
                    torch.from_numpy(tip_coord_normalized.reshape(1, 3)).float().cuda()
                )

                with torch.no_grad():
                    warped_normalized = decoder.warper(
                        tip_tensor, first_appear_time.cuda(), target_time.cuda()
                    )

                # Denormalize
                warped_coord = (
                    warped_normalized[0].cpu().numpy() / scale.numpy()
                ) + mean.numpy()
                warped_coord = warped_coord.flatten()
                warped_tip_coords_list.append(warped_coord)

        # Clean any accumulated gradients after processing each timepoint
        if hasattr(decoder.warper, "clean_grads"):
            decoder.warper.clean_grads()

        warped_tip_coords = np.array(warped_tip_coords_list)

        # Compute EPE for tips that exist in GT at this timepoint
        # We track ALL tips continuously, but only evaluate EPE where GT exists
        num_gt_tips = len(gt_tip_coords)
        num_warped_tips = len(warped_tip_coords)

        # Match warped tips to GT tips (use first N tips that have been initialized)
        # Assumption: tips are initialized in order and GT tips are in the same order
        num_tips_to_eval = min(num_warped_tips, num_gt_tips)

        if num_tips_to_eval > 0:
            # Compute EPE only for tips that have GT at this frame
            warped_to_eval = warped_tip_coords[:num_tips_to_eval]
            gt_to_eval = gt_tip_coords[:num_tips_to_eval]

            # Check if this is the first appearance for any tip
            # Skip EPE computation for tips at their first appearance (like we skip first frame for CD/NC)
            tips_to_include = []
            for tip_id in range(num_tips_to_eval):
                if tip_id in tracked_tips:
                    first_appear_idx, _, _ = tracked_tips[tip_id]
                    if t_idx != first_appear_idx:
                        # Not the first appearance, include this tip
                        tips_to_include.append(tip_id)

            if len(tips_to_include) > 0:
                # Compute EPE only for tips not at their first appearance
                tips_to_include = np.array(tips_to_include)
                epe_per_tip_filtered = np.linalg.norm(
                    warped_to_eval[tips_to_include] - gt_to_eval[tips_to_include],
                    axis=1,
                )
                mean_epe = epe_per_tip_filtered.mean()

                # For split computation, we need EPE per tip (including None for skipped tips)
                epe_per_tip = np.full(num_tips_to_eval, np.nan)
                epe_per_tip[tips_to_include] = np.linalg.norm(
                    warped_to_eval[tips_to_include] - gt_to_eval[tips_to_include],
                    axis=1,
                )
            else:
                # All tips are at their first appearance, skip this frame
                mean_epe = None
                epe_per_tip = np.full(num_tips_to_eval, np.nan)

            epe_values_all.append(mean_epe)

            # Store ALL warped tips (for continuous tracking), but only GT tips that exist
            warped_coords_all.append(warped_tip_coords)  # All tracked tips
            gt_coords_all.append(gt_tip_coords)  # Only GT tips at this frame

            if mean_epe is not None:
                if num_warped_tips != num_gt_tips:
                    logging.info(
                        f"Timepoint {t_idx}: Tracked {num_warped_tips} tips, GT has {num_gt_tips} tips, "
                        f"evaluated {len(tips_to_include)} tips (skipped {num_tips_to_eval - len(tips_to_include)} at first appearance), Mean EPE = {mean_epe:.6f}"
                    )
                else:
                    logging.info(
                        f"Timepoint {t_idx}: Tracked {num_warped_tips} tips, "
                        f"evaluated {len(tips_to_include)} tips (skipped {num_tips_to_eval - len(tips_to_include)} at first appearance), Mean EPE = {mean_epe:.6f}"
                    )
            else:
                logging.info(
                    f"Timepoint {t_idx}: Skipped EPE computation (all {num_tips_to_eval} tips at first appearance)"
                )

            # Handle splits if provided
            if point_splits:
                # Load GT points to check which split tips belong to
                gt_points, _, _ = pcu.load_mesh_vfn(gt_path, dtype=np.float32)
                gt_points, _, _ = pcu.deduplicate_point_cloud(
                    gt_points, 1e-15, return_index=True
                )

                # Map to original timepoint index (important for interpolation mode)
                original_t_idx = original_indices[t_idx]
                timepoint_data = point_splits["timepoints"][original_t_idx]
                train_indices = np.array(
                    timepoint_data["train_indices"], dtype=np.int64
                )
                test_indices = np.array(timepoint_data["test_indices"], dtype=np.int64)

                # Load tip indices from GT
                gt_tip_indices_in_full_cloud, _ = load_leaf_tips_from_ply(gt_path)

                # Separate tips into train and test (only for tips we evaluated)
                train_tip_mask = np.isin(
                    gt_tip_indices_in_full_cloud[:num_tips_to_eval], train_indices
                )
                test_tip_mask = np.isin(
                    gt_tip_indices_in_full_cloud[:num_tips_to_eval], test_indices
                )

                # Compute EPE for train tips
                if train_tip_mask.sum() > 0:
                    train_epe_values = epe_per_tip[train_tip_mask]
                    # Filter out NaN values (which correspond to tips at their first appearance)
                    # These tips were excluded from EPE computation above (lines 703-709)
                    train_epe_values = train_epe_values[~np.isnan(train_epe_values)]
                    if len(train_epe_values) > 0:
                        train_epe = train_epe_values.mean()
                        epe_values_train.append(train_epe)
                    else:
                        epe_values_train.append(None)
                else:
                    epe_values_train.append(None)

                # Compute EPE for test tips
                if test_tip_mask.sum() > 0:
                    test_epe_values = epe_per_tip[test_tip_mask]
                    # Filter out NaN values (which correspond to tips at their first appearance)
                    # These tips were excluded from EPE computation above (lines 703-709)
                    test_epe_values = test_epe_values[~np.isnan(test_epe_values)]
                    if len(test_epe_values) > 0:
                        test_epe = test_epe_values.mean()
                        epe_values_test.append(test_epe)
                    else:
                        epe_values_test.append(None)
                    # Store ALL tracked test tips for AccS/AccR (not just evaluated ones)
                    # This allows continuous flow computation even when GT is missing in some frames
                    warped_coords_test.append(warped_tip_coords)
                    gt_coords_test.append(gt_tip_coords)
                else:
                    epe_values_test.append(None)
                    # Still store warped coords for continuous tracking, but no GT
                    warped_coords_test.append(warped_tip_coords)
                    gt_coords_test.append(None)
        else:
            logging.warning(
                f"Timepoint {t_idx}: No tips to evaluate (warped: {num_warped_tips}, GT: {num_gt_tips})"
            )
            epe_values_all.append(None)
            # Still store warped coords for continuous tracking
            warped_coords_all.append(warped_tip_coords if num_warped_tips > 0 else None)
            gt_coords_all.append(None)
            if point_splits:
                epe_values_train.append(None)
                epe_values_test.append(None)
                warped_coords_test.append(
                    warped_tip_coords if num_warped_tips > 0 else None
                )
                gt_coords_test.append(None)

    result = {
        "all": epe_values_all,
        "warped_coords_all": warped_coords_all,
        "gt_coords_all": gt_coords_all,
    }
    if point_splits:
        result["train"] = epe_values_train
        result["test"] = epe_values_test
        result["warped_coords_test"] = warped_coords_test
        result["gt_coords_test"] = gt_coords_test

        # Summary logging
        train_valid = sum(1 for v in epe_values_train if v is not None)
        test_valid = sum(1 for v in epe_values_test if v is not None)
        logging.info(f"=== TIP TRACKING SUMMARY ===")
        logging.info(f"Total unique tips tracked: {len(tracked_tips)}")
        logging.info(
            f"Timesteps with valid TRAIN tip EPE: {train_valid}/{len(epe_values_train)}"
        )
        logging.info(
            f"Timesteps with valid TEST tip EPE: {test_valid}/{len(epe_values_test)}"
        )

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
    gt_paths, gt_files, original_indices = get_ground_truth_paths(args)

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

    # For nearest neighbor EPE fallback, we need full predicted point clouds
    # (not split into train/test)
    pred_points_full_list = []

    # Load all meshes and compute per-timestep metrics
    for i, (pred_file, gt_path) in enumerate(zip(pred_files, gt_paths)):
        pred_path = os.path.join(pred_mesh_dir, pred_file)

        logging.info(f"Processing timestep {i+1}/{len(pred_files)}: {pred_file}")

        # Load ground truth point cloud (and normals if needed) FIRST
        # We need to know the GT size to determine how many points to sample from prediction
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

        # Determine number of points to sample from prediction
        if point_splits:
            # If using point splits, we need to sample enough points to match GT test set size
            # Map to original timepoint index (important for interpolation mode)
            original_i = original_indices[i]
            timepoint_data = point_splits["timepoints"][original_i]
            train_indices = np.array(timepoint_data["train_indices"], dtype=np.int64)
            test_indices = np.array(timepoint_data["test_indices"], dtype=np.int64)

            # Apply splits to GT points to get the actual counts
            gt_points_train = apply_point_split(gt_points, train_indices)
            gt_points_test = apply_point_split(gt_points, test_indices)

            # Sample from prediction: num_train_points + num_test_points
            num_pred_samples = len(gt_points_train) + len(gt_points_test)
        else:
            # No point splits, use default num_samples
            num_pred_samples = args.num_samples

        # Load predicted mesh as points (and normals if needed)
        try:
            if args.compute_normal_consistency:
                pred_points, pred_normals = load_mesh_as_points(
                    pred_path, num_pred_samples, return_normals=True
                )
            else:
                pred_points = load_mesh_as_points(
                    pred_path, num_pred_samples, return_normals=False
                )
                pred_normals = None
        except Exception as e:
            logging.error(f"Error loading predicted mesh {pred_file}: {e}")
            continue

        # Save full predicted points for nearest neighbor EPE computation
        pred_points_full_list.append(pred_points)

        # Apply point splits if provided
        if point_splits:
            # For predicted points, we use the same split indices as GT
            # This assumes the sampled points maintain the same index correspondence
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

                pred_normals_train_tensor = torch.tensor(
                    pred_normals_train, dtype=torch.float32
                ).cuda()
                gt_normals_train_tensor = torch.tensor(
                    gt_normals_train, dtype=torch.float32
                ).cuda()
                pred_normals_test_tensor = torch.tensor(
                    pred_normals_test, dtype=torch.float32
                ).cuda()
                gt_normals_test_tensor = torch.tensor(
                    gt_normals_test, dtype=torch.float32
                ).cuda()

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
                gt_tip_indices_in_full_cloud, gt_tip_coords = load_leaf_tips_from_ply(
                    gt_path
                )

                if len(gt_tip_indices_in_full_cloud) > 0:

                    # Separate tips into train and test based on which split they fall into
                    train_tip_mask = np.isin(
                        gt_tip_indices_in_full_cloud, train_indices
                    )
                    test_tip_mask = np.isin(gt_tip_indices_in_full_cloud, test_indices)

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
                pred_normals_tensor = torch.tensor(
                    pred_normals, dtype=torch.float32
                ).cuda()
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
    # If using point splits, num_samples_per_mesh will vary per timestep to match GT
    # Otherwise, it's the fixed args.num_samples
    metrics_json = {
        "epoch": epoch_num,
        "num_samples_per_mesh": (
            args.num_samples
            if not point_splits
            else "variable (matches GT point count)"
        ),
        "has_point_splits": point_splits is not None,
    }

    if point_splits:
        # Compute separate train and test metrics
        logging.info("Computing separate train and test metrics...")
        logging.info("Skipping first frame (index 0) for CD/NC computation")

        # Train metrics - Chamfer Distance and Normal Consistency using PyTorch3D
        # Skip first frame (index 0)
        cd_train_values = []
        nc_train_values = []

        if args.compute_normal_consistency:
            # Compute both CD and NC together, skip first frame
            for i, (pred_pts, pred_norms, gt_pts, gt_norms) in enumerate(
                zip(
                    pred_points_train_list,
                    pred_normals_train_list,
                    gt_points_train_list,
                    gt_normals_train_list,
                )
            ):
                if i == 0:
                    continue  # Skip first frame
                cd, nc = compute_pytorch3d_chamfer_distance(
                    pred_pts, gt_pts, pred_norms, gt_norms
                )
                cd_train_values.append(cd)
                nc_train_values.append(nc)
        else:
            # Compute only CD, skip first frame
            for i, (pred_pts, gt_pts) in enumerate(
                zip(pred_points_train_list, gt_points_train_list)
            ):
                if i == 0:
                    continue  # Skip first frame
                cd, _ = compute_pytorch3d_chamfer_distance(pred_pts, gt_pts)
                cd_train_values.append(cd)

        cd_train_tensor = torch.tensor(cd_train_values)
        metrics_json["train"] = {
            "num_timesteps": len(pred_points_train_list),
            "mean_chamfer_distance": float(cd_train_tensor.mean().item()),
            "std_chamfer_distance": float(cd_train_tensor.std().item()),
            "chamfer_distances_per_timestep": [float(cd) for cd in cd_train_values],
        }

        if args.compute_normal_consistency and len(nc_train_values) > 0:
            nc_train_tensor = torch.tensor(nc_train_values)
            metrics_json["train"]["mean_normal_consistency"] = float(
                nc_train_tensor.mean().item()
            )
            metrics_json["train"]["std_normal_consistency"] = float(
                nc_train_tensor.std().item()
            )
            metrics_json["train"]["normal_consistency_per_timestep"] = [
                float(nc) for nc in nc_train_values
            ]

        # Test metrics - Chamfer Distance and Normal Consistency using PyTorch3D
        # Skip first frame (index 0)
        cd_test_values = []
        nc_test_values = []

        if args.compute_normal_consistency:
            # Compute both CD and NC together, skip first frame
            for i, (pred_pts, pred_norms, gt_pts, gt_norms) in enumerate(
                zip(
                    pred_points_test_list,
                    pred_normals_test_list,
                    gt_points_test_list,
                    gt_normals_test_list,
                )
            ):
                if i == 0:
                    continue  # Skip first frame
                cd, nc = compute_pytorch3d_chamfer_distance(
                    pred_pts, gt_pts, pred_norms, gt_norms
                )
                cd_test_values.append(cd)
                nc_test_values.append(nc)
        else:
            # Compute only CD, skip first frame
            for i, (pred_pts, gt_pts) in enumerate(
                zip(pred_points_test_list, gt_points_test_list)
            ):
                if i == 0:
                    continue  # Skip first frame
                cd, _ = compute_pytorch3d_chamfer_distance(pred_pts, gt_pts)
                cd_test_values.append(cd)

        cd_test_tensor = torch.tensor(cd_test_values)
        metrics_json["test"] = {
            "num_timesteps": len(pred_points_test_list),
            "mean_chamfer_distance": float(cd_test_tensor.mean().item()),
            "std_chamfer_distance": float(cd_test_tensor.std().item()),
            "chamfer_distances_per_timestep": [float(cd) for cd in cd_test_values],
        }

        if args.compute_normal_consistency and len(nc_test_values) > 0:
            nc_test_tensor = torch.tensor(nc_test_values)
            metrics_json["test"]["mean_normal_consistency"] = float(
                nc_test_tensor.mean().item()
            )
            metrics_json["test"]["std_normal_consistency"] = float(
                nc_test_tensor.std().item()
            )
            metrics_json["test"]["normal_consistency_per_timestep"] = [
                float(nc) for nc in nc_test_values
            ]

        # Compute EPE for leaf tips using warper-based tracking
        if args.compute_tip_epe:
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
            decoder, scale, mean, normalized_times = load_decoder_model(
                args, specs, sdf_dataset, epoch_num
            )

            # Ensure gradient computation is disabled for tracking
            decoder.warper.turoff_grad()
            if hasattr(decoder.warper, "clean_grads"):
                decoder.warper.clean_grads()

            # Track tips with warper
            tip_epe_results = track_tips_with_warper(
                decoder,
                gt_paths,
                sdf_dataset,
                scale,
                mean,
                normalized_times,
                point_splits,
                original_indices,
            )

            # Clean accumulated gradients after tracking
            if hasattr(decoder.warper, "clean_grads"):
                decoder.warper.clean_grads()

            # Add results to metrics_json
            # Train tips
            train_epe_values = [v for v in tip_epe_results["train"] if v is not None]
            if len(train_epe_values) > 0:
                train_epe_tensor = torch.tensor(train_epe_values)
                metrics_json["train"]["num_timesteps_with_tips"] = len(train_epe_values)
                metrics_json["train"]["mean_tip_epe"] = float(
                    train_epe_tensor.mean().item()
                )
                std_val = float(train_epe_tensor.std().item())
                metrics_json["train"]["std_tip_epe"] = (
                    None if (np.isnan(std_val) or np.isinf(std_val)) else std_val
                )
                metrics_json["train"]["tip_epe_per_timestep"] = [
                    float(v) for v in tip_epe_results["train"] if v is not None
                ]
                logging.info(
                    f"Train Set - Leaf Tip EPE (warper): {metrics_json['train']['mean_tip_epe']:.6f}"
                )
            else:
                metrics_json["train"]["num_timesteps_with_tips"] = 0
                metrics_json["train"]["mean_tip_epe"] = None
                metrics_json["train"]["std_tip_epe"] = None
                metrics_json["train"]["tip_epe_per_timestep"] = []
                logging.info(
                    "Train Set - No leaf tips found (expected if tips are in test set)"
                )

            # Test tips
            test_epe_values = [v for v in tip_epe_results["test"] if v is not None]
            if len(test_epe_values) > 0:
                test_epe_tensor = torch.tensor(test_epe_values)
                metrics_json["test"]["num_timesteps_with_tips"] = len(test_epe_values)
                metrics_json["test"]["mean_tip_epe"] = float(
                    test_epe_tensor.mean().item()
                )
                std_val = float(test_epe_tensor.std().item())
                metrics_json["test"]["std_tip_epe"] = (
                    None if (np.isnan(std_val) or np.isinf(std_val)) else std_val
                )
                metrics_json["test"]["tip_epe_per_timestep"] = [
                    float(v) for v in tip_epe_results["test"] if v is not None
                ]
                logging.info(
                    f"Test Set - Leaf Tip EPE (warper): {metrics_json['test']['mean_tip_epe']:.6f}"
                )
            else:
                metrics_json["test"]["num_timesteps_with_tips"] = 0
                metrics_json["test"]["mean_tip_epe"] = None
                metrics_json["test"]["std_tip_epe"] = None
                metrics_json["test"]["tip_epe_per_timestep"] = []
                logging.info("Test Set - No leaf tips found")

        # Log results
        logging.info(
            f"Train Set - Mean CD: {metrics_json['train']['mean_chamfer_distance']:.6f}"
        )
        logging.info(
            f"Train Set - Std CD: {metrics_json['train']['std_chamfer_distance']:.6f}"
        )
        if (
            args.compute_normal_consistency
            and "mean_normal_consistency" in metrics_json["train"]
        ):
            logging.info(
                f"Train Set - Mean NC: {metrics_json['train']['mean_normal_consistency']:.6f}"
            )
            logging.info(
                f"Train Set - Std NC: {metrics_json['train']['std_normal_consistency']:.6f}"
            )

        logging.info(
            f"Test Set - Mean CD: {metrics_json['test']['mean_chamfer_distance']:.6f}"
        )
        logging.info(
            f"Test Set - Std CD: {metrics_json['test']['std_chamfer_distance']:.6f}"
        )
        if (
            args.compute_normal_consistency
            and "mean_normal_consistency" in metrics_json["test"]
        ):
            logging.info(
                f"Test Set - Mean NC: {metrics_json['test']['mean_normal_consistency']:.6f}"
            )
            logging.info(
                f"Test Set - Std NC: {metrics_json['test']['std_normal_consistency']:.6f}"
            )

    else:
        # No splits - compute metrics on all points (backward compatible)
        logging.info("Computing metrics on full point clouds...")
        logging.info("Skipping first frame (index 0) for CD/NC computation")

        # Chamfer Distance and Normal Consistency using PyTorch3D
        # Skip first frame (index 0)
        cd_values = []
        nc_values = []

        if args.compute_normal_consistency:
            # Compute both CD and NC together, skip first frame
            for i, (pred_pts, pred_norms, gt_pts, gt_norms) in enumerate(
                zip(
                    pred_points_list, pred_normals_list, gt_points_list, gt_normals_list
                )
            ):
                if i == 0:
                    continue  # Skip first frame
                cd, nc = compute_pytorch3d_chamfer_distance(
                    pred_pts, gt_pts, pred_norms, gt_norms
                )
                cd_values.append(cd)
                nc_values.append(nc)
        else:
            # Compute only CD, skip first frame
            for i, (pred_pts, gt_pts) in enumerate(
                zip(pred_points_list, gt_points_list)
            ):
                if i == 0:
                    continue  # Skip first frame
                cd, _ = compute_pytorch3d_chamfer_distance(pred_pts, gt_pts)
                cd_values.append(cd)

        cd_tensor = torch.tensor(cd_values)
        metrics_json["num_timesteps"] = len(pred_points_list)
        metrics_json["mean_chamfer_distance"] = float(cd_tensor.mean().item())
        metrics_json["std_chamfer_distance"] = float(cd_tensor.std().item())
        metrics_json["chamfer_distances_per_timestep"] = [float(cd) for cd in cd_values]

        if args.compute_normal_consistency and len(nc_values) > 0:
            nc_tensor = torch.tensor(nc_values)
            metrics_json["mean_normal_consistency"] = float(nc_tensor.mean().item())
            metrics_json["std_normal_consistency"] = float(nc_tensor.std().item())
            metrics_json["normal_consistency_per_timestep"] = [
                float(nc) for nc in nc_values
            ]

        # Tip EPE using warper-based tracking
        if args.compute_tip_epe:
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

            # Extract interp_stride from experiment name (if present)
            parts = args.experiment_name.split("_")
            interp_stride_for_dataset = None
            for i, part in enumerate(parts):
                if part == "stride" and i + 1 < len(parts):
                    if i > 0 and parts[i - 1] == "interp":
                        interp_stride_for_dataset = int(parts[i + 1])

            sdf_dataset = data_pheno4d.DeformSamples(
                args.object_name,
                start_idx=0,
                gap=None,
                train_step=1,
                use_dense=False,
                split_file=split_file,
                use_train_points=True,  # For dataset normalization, doesn't matter in interpolation mode
                interp_stride=interp_stride_for_dataset,
            )

            # Load decoder model
            decoder, scale, mean, normalized_times = load_decoder_model(
                args, specs, sdf_dataset, epoch_num
            )

            # Ensure gradient computation is disabled for tracking
            decoder.warper.turoff_grad()
            if hasattr(decoder.warper, "clean_grads"):
                decoder.warper.clean_grads()

            # Track tips with warper (no splits)
            tip_epe_results = track_tips_with_warper(
                decoder,
                gt_paths,
                sdf_dataset,
                scale,
                mean,
                normalized_times,
                point_splits=None,
                original_indices=original_indices,
            )

            # Clean accumulated gradients after tracking
            if hasattr(decoder.warper, "clean_grads"):
                decoder.warper.clean_grads()

            # Add results to metrics_json
            epe_values = [v for v in tip_epe_results["all"] if v is not None]
            if len(epe_values) > 0:
                epe_tensor = torch.tensor(epe_values)
                metrics_json["num_timesteps_with_tips"] = len(epe_values)
                metrics_json["mean_tip_epe"] = float(epe_tensor.mean().item())
                std_val = float(epe_tensor.std().item())
                metrics_json["std_tip_epe"] = (
                    None if (np.isnan(std_val) or np.isinf(std_val)) else std_val
                )
                metrics_json["tip_epe_per_timestep"] = [
                    float(v) for v in tip_epe_results["all"] if v is not None
                ]
                logging.info(
                    f"Leaf Tip EPE (warper): {metrics_json['mean_tip_epe']:.6f}"
                )
            else:
                metrics_json["num_timesteps_with_tips"] = 0
                metrics_json["mean_tip_epe"] = None
                metrics_json["std_tip_epe"] = None
                metrics_json["tip_epe_per_timestep"] = []
                logging.info("No leaf tips found in any timestep")

        # Log results
        logging.info(
            f"Mean Chamfer Distance: {metrics_json['mean_chamfer_distance']:.6f}"
        )
        logging.info(
            f"Std Chamfer Distance: {metrics_json['std_chamfer_distance']:.6f}"
        )
        if args.compute_normal_consistency:
            logging.info(
                f"Mean Normal Consistency: {metrics_json['mean_normal_consistency']:.6f}"
            )
            logging.info(
                f"Std Normal Consistency: {metrics_json['std_normal_consistency']:.6f}"
            )
        if (
            args.compute_tip_epe
            and "mean_tip_epe" in metrics_json
            and metrics_json["mean_tip_epe"] is not None
        ):
            logging.info(f"Mean Tip EPE: {metrics_json['mean_tip_epe']:.6f}")
            logging.info(f"Std Tip EPE: {metrics_json['std_tip_epe']:.6f}")

    # Compute scene flow metrics (AccS, AccR, EPE3D) on leaf tips using warped coordinates
    # This uses the same warped tip tracking that we use for EPE computation
    logging.info("Computing accuracy metrics (AccS, AccR) on tracked leaf tips...")

    try:
        if point_splits:
            # Use warped test tip coordinates from tip tracking
            if args.compute_tip_epe and "warped_coords_test" in tip_epe_results:
                warped_coords_test = tip_epe_results["warped_coords_test"]
                gt_coords_test = tip_epe_results["gt_coords_test"]

                test_accs_values = []
                test_accr_values = []

                # Compute position-based accuracy at each timestep
                # Skip first appearance (same logic as EPE computation)
                for i in range(len(warped_coords_test)):
                    if warped_coords_test[i] is None or gt_coords_test[i] is None:
                        continue

                    num_warped_tips = len(warped_coords_test[i])
                    num_gt_tips = len(gt_coords_test[i])
                    num_tips_to_eval = min(num_warped_tips, num_gt_tips)

                    if num_tips_to_eval == 0:
                        continue

                    # Skip first appearance (consistent with EPE computation)
                    # Check if any tips at this timestep are at their first appearance
                    # We need to access tracked_tips from tip tracking, but we don't have it here
                    # Instead, we'll skip timestep 0 if it contains tips (simpler approach)
                    # This works because tips appear sequentially and the first tip appears at t=0
                    if i == 0:
                        # Check if this is the first timestep with tips
                        has_earlier_tips = False
                        for j in range(i):
                            if (
                                warped_coords_test[j] is not None
                                and gt_coords_test[j] is not None
                            ):
                                if (
                                    len(warped_coords_test[j]) > 0
                                    and len(gt_coords_test[j]) > 0
                                ):
                                    has_earlier_tips = True
                                    break
                        if not has_earlier_tips:
                            # This is first appearance, skip it
                            continue

                    # Compute position errors (same as EPE)
                    warped_pos = warped_coords_test[i][:num_tips_to_eval]
                    gt_pos = gt_coords_test[i][:num_tips_to_eval]

                    # Convert to tensors
                    warped_tensor = torch.from_numpy(warped_pos).float()
                    gt_tensor = torch.from_numpy(gt_pos).float()

                    # Compute AccS and AccR using position errors (not flow)
                    # Note: EPE3D from scene_flow_metrics is redundant with mean_tip_epe, so we ignore it
                    _, accs, accr = scene_flow_metrics(warped_tensor, gt_tensor)
                    test_accs_values.append(accs)
                    test_accr_values.append(accr)

                if len(test_accs_values) > 0:
                    metrics_json["test"]["mean_AccS"] = float(np.mean(test_accs_values))
                    metrics_json["test"]["std_AccS"] = float(np.std(test_accs_values))
                    metrics_json["test"]["AccS_per_timestep"] = [
                        float(v) for v in test_accs_values
                    ]

                    metrics_json["test"]["mean_AccR"] = float(np.mean(test_accr_values))
                    metrics_json["test"]["std_AccR"] = float(np.std(test_accr_values))
                    metrics_json["test"]["AccR_per_timestep"] = [
                        float(v) for v in test_accr_values
                    ]

                    logging.info(
                        f"Test Set - Mean AccS (leaf tips): {metrics_json['test']['mean_AccS']:.2f}%"
                    )
                    logging.info(
                        f"Test Set - Mean AccR (leaf tips): {metrics_json['test']['mean_AccR']:.2f}%"
                    )
                else:
                    logging.warning("No valid timesteps for AccS/AccR computation")
            else:
                logging.warning(
                    "Warped tip coordinates not available, skipping AccS/AccR on tips"
                )
        else:
            # No splits - use all warped coordinates
            if args.compute_tip_epe and "warped_coords_all" in tip_epe_results:
                warped_coords_all = tip_epe_results["warped_coords_all"]
                gt_coords_all = tip_epe_results["gt_coords_all"]

                accs_values = []
                accr_values = []

                # Compute position-based accuracy at each timestep
                # Skip first appearance (same logic as EPE computation)
                for i in range(len(warped_coords_all)):
                    if warped_coords_all[i] is None or gt_coords_all[i] is None:
                        continue

                    num_warped_tips = len(warped_coords_all[i])
                    num_gt_tips = len(gt_coords_all[i])
                    num_tips_to_eval = min(num_warped_tips, num_gt_tips)

                    if num_tips_to_eval == 0:
                        continue

                    # Skip first appearance (consistent with EPE computation)
                    if i == 0:
                        # Check if this is the first timestep with tips
                        has_earlier_tips = False
                        for j in range(i):
                            if (
                                warped_coords_all[j] is not None
                                and gt_coords_all[j] is not None
                            ):
                                if (
                                    len(warped_coords_all[j]) > 0
                                    and len(gt_coords_all[j]) > 0
                                ):
                                    has_earlier_tips = True
                                    break
                        if not has_earlier_tips:
                            # This is first appearance, skip it
                            continue

                    # Compute position errors (same as EPE)
                    warped_pos = warped_coords_all[i][:num_tips_to_eval]
                    gt_pos = gt_coords_all[i][:num_tips_to_eval]

                    # Convert to tensors
                    warped_tensor = torch.from_numpy(warped_pos).float()
                    gt_tensor = torch.from_numpy(gt_pos).float()

                    # Compute AccS and AccR using position errors (not flow)
                    # Note: EPE3D from scene_flow_metrics is redundant with mean_tip_epe, so we ignore it
                    _, accs, accr = scene_flow_metrics(warped_tensor, gt_tensor)
                    accs_values.append(accs)
                    accr_values.append(accr)

                if len(accs_values) > 0:
                    metrics_json["mean_AccS"] = float(np.mean(accs_values))
                    metrics_json["std_AccS"] = float(np.std(accs_values))
                    metrics_json["AccS_per_timestep"] = [float(v) for v in accs_values]

                    metrics_json["mean_AccR"] = float(np.mean(accr_values))
                    metrics_json["std_AccR"] = float(np.std(accr_values))
                    metrics_json["AccR_per_timestep"] = [float(v) for v in accr_values]

                    logging.info(
                        f"Mean AccS (leaf tips): {metrics_json['mean_AccS']:.2f}%"
                    )
                    logging.info(
                        f"Mean AccR (leaf tips): {metrics_json['mean_AccR']:.2f}%"
                    )
                else:
                    logging.warning("No valid timesteps for AccS/AccR computation")
            else:
                logging.warning(
                    "Warped tip coordinates not available, skipping AccS/AccR on tips"
                )
    except Exception as e:
        import traceback

        logging.error(f"Error computing scene flow metrics on tips: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        logging.warning("Skipping AccS/AccR computation on tips")

    # Add storage and runtime info
    epoch_str = str(epoch_num)
    storage_mb = compute_storage_size(args.experiment_name, epoch_str)
    runtime_mins = get_runtime_from_wandb(args.experiment_name)

    metrics_json["storage_mb"] = float(storage_mb)
    metrics_json["runtime_minutes"] = (
        float(runtime_mins) if runtime_mins is not None else None
    )

    logging.info(f"Storage size: {storage_mb:.2f} MB")
    if runtime_mins is not None:
        logging.info(
            f"Training runtime: {runtime_mins:.2f} minutes ({runtime_mins/60:.2f} hours)"
        )

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
