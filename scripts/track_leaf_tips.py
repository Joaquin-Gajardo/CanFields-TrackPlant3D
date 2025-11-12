#!/usr/bin/env python3
"""
Track leaf tips through time using the learned deformation field (warper).

This script:
1. Loads a trained CanFields model
2. Extracts leaf tip positions from the ground truth at a reference timepoint
3. Uses the warper to track tips through time
4. Computes EPE between tracked tips and ground truth tips at each timepoint

Usage:
    python scripts/track_leaf_tips.py \
        -d pheno4d \
        -o Pheno4D/processed/v2/organ_sequences/maize/maize_control_plant2 \
        -en results/pheno4d/ode_solve/maize/maize_control_plant2/20251111_052644_maize_control_plant2_P4D_full_l2_0.05_elas_0.05_prob_0.01_start_id_0_gap_9_stride_1 \
        -e 15000 \
        --ref_timepoint 4
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import open3d as o3d
import torch

from datasets import data_pheno4d
from multi_tools import load_dataset
from networks.multi_velocity_debug import Decoder
from utils import add_common_args, configure_logging, load_latent_vectors, load_model_parameters


def build_args():
    parser = argparse.ArgumentParser(
        description="Track leaf tips through time using learned deformation field"
    )
    parser.add_argument(
        "--object_name",
        "-o",
        dest="object_name",
        required=True,
        help="Path to the object sequence (e.g., Pheno4D/processed/v2/organ_sequences/maize/maize_control_plant2)",
    )
    parser.add_argument(
        "--experiment_name",
        "-en",
        dest="experiment_name",
        required=True,
        help="Path to experiment directory",
    )
    parser.add_argument(
        "--dataset_name",
        "-d",
        dest="dataset_name",
        required=True,
        help="Dataset name: 'pheno4d'",
    )
    parser.add_argument(
        "--epoch",
        "-e",
        dest="epoch",
        required=True,
        help="Epoch to evaluate (e.g., 15000)",
    )
    parser.add_argument(
        "--ref_timepoint",
        type=int,
        default=0,
        help="Reference timepoint index to track from (default: 0 = first timepoint)",
    )
    parser.add_argument(
        "--specifications",
        "-s",
        dest="specifications",
        default="configs/specs.json",
        help="Path to specs JSON file",
    )
    parser.add_argument(
        "--point_split_file",
        default=None,
        help="Path to point split JSON file (optional)",
    )

    add_common_args(parser)
    args = parser.parse_args()
    configure_logging(args)
    return args


def load_leaf_tips_from_ply(ply_path: str):
    """Load leaf tip indices and coordinates from a PLY file."""
    try:
        pcd = o3d.t.io.read_point_cloud(ply_path)
        points = pcd.point.positions.numpy()

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


def main():
    args = build_args()

    logging.info("=" * 80)
    logging.info("Tracking Leaf Tips Through Time")
    logging.info("=" * 80)

    # Load specs (resolve path relative to script directory)
    specs_path = args.specifications
    if not os.path.isabs(specs_path):
        # Make relative paths relative to the CanFields root directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        canfields_root = os.path.dirname(script_dir)
        specs_path = os.path.join(canfields_root, specs_path)

    specs = json.load(open(specs_path))

    # Auto-detect split file if not provided
    split_file = args.point_split_file
    if args.dataset_name == "pheno4d" and split_file is None:
        sequence_name = args.object_name.split("/")[-1]
        split_file = f"configs/point_splits/{sequence_name}_split.json"
        if os.path.exists(split_file):
            logging.info(f"Auto-detected point split file: {split_file}")
        else:
            logging.warning(f"Point split file not found: {split_file}")
            split_file = None

    # Load dataset
    logging.info("Loading dataset...")
    sdf_dataset = data_pheno4d.DeformSamples(
        args.object_name,
        start_idx=0,
        gap=None,
        train_step=1,
        use_dense=False,
        split_file=split_file,
        use_train_points=True,
    )

    num_timepoints = len(sdf_dataset)
    logging.info(f"Dataset has {num_timepoints} timepoints")

    # Determine reference timepoint
    ref_idx = args.ref_timepoint

    if ref_idx >= num_timepoints:
        raise ValueError(f"Reference timepoint {ref_idx} out of range (0-{num_timepoints-1})")

    logging.info(f"Tracking tips starting from timepoint {ref_idx}")

    # Get normalization parameters
    scale = sdf_dataset.scale
    mean = sdf_dataset.mean

    # Load model
    logging.info("Loading trained model...")
    latent_size = specs["CodeLength"]

    # Import adjustment network
    from networks.multi_velocity_debug import adjust_net

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

    start_end_vecs = torch.nn.Embedding(2, 128, max_norm=1.0).cuda()
    decoder.warper.equip_embedding(start_end_vecs)

    # Load model as ModuleList (adjust + decoder)
    experiment_directory = args.experiment_name
    epoch = args.epoch

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

    ref_time = sdf_dataset.normalized_times[ref_idx]
    logging.info(f"Reference time: {ref_time.item():.4f}")

    # Load reference leaf tips from GT
    ref_ply_path = os.path.join(
        args.object_name,
        "sparse_pc",
        sdf_dataset.sub_files[ref_idx]
    )

    logging.info(f"Loading reference tips from: {ref_ply_path}")
    ref_tip_indices, ref_tip_coords = load_leaf_tips_from_ply(ref_ply_path)

    if len(ref_tip_coords) == 0:
        logging.error("No leaf tips found in reference timepoint!")
        return

    logging.info(f"Found {len(ref_tip_coords)} leaf tips at reference timepoint")
    logging.info(f"Tip indices in full point cloud: {ref_tip_indices}")

    # Normalize reference tip coordinates
    ref_tip_coords_normalized = (ref_tip_coords - mean.numpy()) * scale.numpy()
    ref_tip_coords_tensor = torch.tensor(ref_tip_coords_normalized, dtype=torch.float32).cuda()

    # Track tips through all timepoints
    # Strategy: dynamically add new tips as they appear
    logging.info("\n" + "=" * 80)
    logging.info("Tracking tips through time...")
    logging.info(f"Strategy: Track existing tips + add new tips as they appear")
    logging.info("=" * 80)

    # Dictionary to track each tip: {tip_id: (first_appearance_idx, first_appearance_coords, first_appearance_time)}
    tracked_tips = {}

    # Initialize with reference tips
    for i, coord in enumerate(ref_tip_coords):
        tracked_tips[i] = (ref_idx, coord, ref_time)

    all_epe_values = []

    for t_idx in range(num_timepoints):
        target_time = sdf_dataset.normalized_times[t_idx]
        target_ply_path = os.path.join(
            args.object_name,
            "sparse_pc",
            sdf_dataset.sub_files[t_idx]
        )

        # Load GT tips at target time
        _, gt_tip_coords = load_leaf_tips_from_ply(target_ply_path)

        if len(gt_tip_coords) == 0:
            logging.info(f"Timepoint {t_idx}: No GT tips found, skipping")
            all_epe_values.append(None)
            continue

        # Check if new tips appeared
        num_current_tips = len(tracked_tips)
        num_gt_tips = len(gt_tip_coords)

        if num_gt_tips > num_current_tips:
            # New tips appeared! Add them to tracking
            num_new_tips = num_gt_tips - num_current_tips
            logging.info(f"Timepoint {t_idx}: Detected {num_new_tips} new tips (total GT: {num_gt_tips}, tracking: {num_current_tips})")

            # Add new tips starting from this timepoint
            for i in range(num_new_tips):
                new_tip_id = num_current_tips + i
                new_tip_coord = gt_tip_coords[new_tip_id]
                tracked_tips[new_tip_id] = (t_idx, new_tip_coord, target_time)
                logging.info(f"  Added tip {new_tip_id} starting from timepoint {t_idx}")

        # Warp all tracked tips to target time
        warped_tip_coords_list = []

        for tip_id in sorted(tracked_tips.keys()):
            first_appear_idx, first_appear_coord, first_appear_time = tracked_tips[tip_id]

            if t_idx < first_appear_idx:
                # Tip hasn't appeared yet, skip
                continue
            elif t_idx == first_appear_idx:
                # Tip just appeared, no warping needed
                warped_tip_coords_list.append(first_appear_coord)
            else:
                # Warp from first appearance to target time
                tip_coord_normalized = (first_appear_coord - mean.numpy()) * scale.numpy()
                # Convert to numpy array first, then to tensor with shape (1, 3)
                tip_coord_np = np.array(tip_coord_normalized).reshape(1, 3)
                tip_tensor = torch.from_numpy(tip_coord_np).float().cuda()

                with torch.no_grad():
                    warped_normalized = decoder.warper(
                        tip_tensor,
                        first_appear_time.cuda(),
                        target_time.cuda()
                    )

                # Denormalize and extract the single point (remove batch dimension)
                # warped_normalized is (1, 3), take [0] to get (3,)
                warped_coord = (warped_normalized[0].cpu().numpy() / scale.numpy()) + mean.numpy()
                # Ensure it's 1D array of shape (3,)
                warped_coord = warped_coord.flatten()
                warped_tip_coords_list.append(warped_coord)

        warped_tip_coords = np.array(warped_tip_coords_list)

        # Match warped tips to GT tips (should have same count now)
        if len(warped_tip_coords) != len(gt_tip_coords):
            logging.warning(
                f"Timepoint {t_idx}: Unexpected tip count mismatch "
                f"(warped: {len(warped_tip_coords)}, GT: {len(gt_tip_coords)})"
            )
            # Take minimum for safety
            min_tips = min(len(warped_tip_coords), len(gt_tip_coords))
            warped_tip_coords = warped_tip_coords[:min_tips]
            gt_tip_coords = gt_tip_coords[:min_tips]

        # Compute EPE
        epe_per_tip = np.linalg.norm(warped_tip_coords - gt_tip_coords, axis=1)
        mean_epe = epe_per_tip.mean()
        all_epe_values.append(mean_epe)

        logging.info(
            f"Timepoint {t_idx} (t={target_time.item():.4f}): "
            f"Tracking {len(warped_tip_coords)} tips, Mean EPE = {mean_epe:.6f}"
        )
        logging.info(f"  Per-tip EPE: {epe_per_tip}")

    # Compute overall statistics
    valid_epe = [epe for epe in all_epe_values if epe is not None]

    if len(valid_epe) > 0:
        logging.info("\n" + "=" * 80)
        logging.info("Summary Statistics")
        logging.info("=" * 80)
        logging.info(f"Valid timepoints: {len(valid_epe)}/{num_timepoints}")
        logging.info(f"Mean EPE: {np.mean(valid_epe):.6f}")
        logging.info(f"Std EPE: {np.std(valid_epe):.6f}")
        logging.info(f"Min EPE: {np.min(valid_epe):.6f}")
        logging.info(f"Max EPE: {np.max(valid_epe):.6f}")

        # Save results
        results = {
            "experiment": args.experiment_name,
            "epoch": args.epoch,
            "reference_timepoint": ref_idx,
            "num_tips": len(ref_tip_coords),
            "epe_per_timepoint": [float(e) if e is not None else None for e in all_epe_values],
            "mean_epe": float(np.mean(valid_epe)),
            "std_epe": float(np.std(valid_epe)),
            "min_epe": float(np.min(valid_epe)),
            "max_epe": float(np.max(valid_epe)),
        }

        output_dir = os.path.join(experiment_directory, "metrics")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"tip_tracking_epoch_{epoch}.json")

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logging.info(f"\nResults saved to: {output_file}")
    else:
        logging.warning("No valid EPE values computed!")


if __name__ == "__main__":
    main()
