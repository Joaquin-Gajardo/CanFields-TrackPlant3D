#!/usr/bin/env python3
"""
Quick test script to manually calculate Chamfer Distance and EPE metrics.

Usage:
    python scripts/test_metrics.py \
        --pred results/.../organs/leaf_1/post_finetune/predicted/pred_*.ply \
        --gt results/.../organs/leaf_1/post_finetune/ground_truth/gt_*.ply

This will compute:
- Train CD (Chamfer Distance on train points)
- Test CD (Chamfer Distance on test points)
- Test EPE (End Point Error on leaf tip, test set only)
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from pytorch3d.loss import chamfer_distance as pytorch3d_chamfer_distance


def chamfer_distance_numpy(src, tgt):
    """
    Compute one-way Chamfer Distance from src to tgt using numpy.

    """
    squared_distances = ((src[:, None, :] - tgt[None, :, :]) ** 2).sum(axis=2)
    min_distances = squared_distances.min(axis=1)
    return min_distances.mean()


def chamfer_distance_numpy_symmetric(pred, gt):
    """Compute symmetric Chamfer Distance using numpy (sum of both directions)."""
    cd_pred_to_gt = chamfer_distance_numpy(pred, gt)
    cd_gt_to_pred = chamfer_distance_numpy(gt, pred)
    return cd_pred_to_gt + cd_gt_to_pred


def chamfer_distance_pytorch3d(pred, gt):
    """
    Compute Chamfer Distance using PyTorch3D (same as used in training).

    Args:
        pred: (N, 3) numpy array
        gt: (M, 3) numpy array

    Returns:
        float: Chamfer distance
    """
    # Convert to torch tensors and add batch dimension
    pred_tensor = torch.from_numpy(pred).float().unsqueeze(0)  # (1, N, 3)
    gt_tensor = torch.from_numpy(gt).float().unsqueeze(0)  # (1, M, 3)

    # Compute Chamfer Distance
    cd, _ = pytorch3d_chamfer_distance(pred_tensor, gt_tensor)

    return cd.item()


def load_ply(path):
    """Load PLY file and extract points, is_train, and is_leaf_tip."""
    pcd = o3d.t.io.read_point_cloud(str(path))

    points = pcd.point.positions.numpy()

    # Extract is_train if available
    is_train = None
    if "is_train" in pcd.point:
        is_train = pcd.point.is_train.numpy().flatten()

    # Extract is_leaf_tip if available
    is_leaf_tip = None
    if "is_leaf_tip" in pcd.point:
        is_leaf_tip = pcd.point.is_leaf_tip.numpy().flatten()

    return points, is_train, is_leaf_tip


def main(args):
    print("=" * 80)
    print("Manual Metrics Calculation")
    print("=" * 80)
    print(f"Predicted: {args.pred}")
    print(f"Ground Truth: {args.gt}")
    print()

    # Load point clouds
    pred_points, pred_is_train, pred_is_tip = load_ply(args.pred)
    gt_points, gt_is_train, gt_is_tip = load_ply(args.gt)

    print(f"Predicted cloud: {len(pred_points)} points")
    print(f"Ground truth cloud: {len(gt_points)} points")

    # Check if is_train attribute exists
    if pred_is_train is None or gt_is_train is None:
        print("\n⚠️  Warning: is_train attribute not found in one or both PLY files")
        print("Computing overall CD only (no train/test split)")

        # Compute overall CD
        cd_overall_numpy = chamfer_distance_numpy_symmetric(pred_points, gt_points)
        cd_overall_pytorch3d = chamfer_distance_pytorch3d(pred_points, gt_points)
        print(f"\n📊 Overall Chamfer Distance:")
        print(f"   NumPy:     {cd_overall_numpy:.6f}")
        print(f"   PyTorch3D: {cd_overall_pytorch3d:.6f}")

    else:
        print(
            f"  Predicted: {np.sum(pred_is_train == 1)} train, {np.sum(pred_is_train == 0)} test"
        )
        print(
            f"  GT: {np.sum(gt_is_train == 1)} train, {np.sum(gt_is_train == 0)} test"
        )

        # Separate train and test points
        pred_train = pred_points[pred_is_train == 1]
        pred_test = pred_points[pred_is_train == 0]
        gt_train = gt_points[gt_is_train == 1]
        gt_test = gt_points[gt_is_train == 0]

        # Compute train CD
        if len(pred_train) > 0 and len(gt_train) > 0:
            cd_train_numpy = chamfer_distance_numpy_symmetric(pred_train, gt_train)
            cd_train_pytorch3d = chamfer_distance_pytorch3d(pred_train, gt_train)
            print(f"\n📊 Train Chamfer Distance:")
            print(f"   NumPy:     {cd_train_numpy:.6f}")
            print(f"   PyTorch3D: {cd_train_pytorch3d:.6f}")
        else:
            print(f"\n📊 Train Chamfer Distance: N/A (no train points)")

        # Compute test CD
        if len(pred_test) > 0 and len(gt_test) > 0:
            cd_test_numpy = chamfer_distance_numpy_symmetric(pred_test, gt_test)
            cd_test_pytorch3d = chamfer_distance_pytorch3d(pred_test, gt_test)
            print(f"\n📊 Test Chamfer Distance:")
            print(f"   NumPy:      {cd_test_numpy:.6f}")
            print(f"   PyTorch3D:  {cd_test_pytorch3d:.6f}")
        else:
            print(f"\n📊 Test Chamfer Distance: N/A (no test points)")

    # Compute EPE (End Point Error) if leaf tip exists
    if pred_is_tip is not None and gt_is_tip is not None:
        pred_tip_mask = pred_is_tip == 1
        gt_tip_mask = gt_is_tip == 1

        if np.any(pred_tip_mask) and np.any(gt_tip_mask):
            pred_tip_pos = pred_points[pred_tip_mask]
            gt_tip_pos = gt_points[gt_tip_mask]

            # Should be single point, but take first if multiple
            if len(pred_tip_pos) > 1:
                print(
                    f"\n⚠️  Warning: Multiple predicted tips found ({len(pred_tip_pos)}), using first"
                )
                pred_tip_pos = pred_tip_pos[0:1]
            if len(gt_tip_pos) > 1:
                print(
                    f"⚠️  Warning: Multiple GT tips found ({len(gt_tip_pos)}), using first"
                )
                gt_tip_pos = gt_tip_pos[0:1]

            epe = np.linalg.norm(pred_tip_pos[0] - gt_tip_pos[0])
            print(f"\n📊 Leaf Tip EPE: {epe:.4f}")

            # Check if tip is in train or test set
            if pred_is_train is not None:
                tip_idx = np.where(pred_tip_mask)[0][0]
                tip_in_train = pred_is_train[tip_idx] == 1
                tip_in_test = pred_is_train[tip_idx] == 0
                if tip_in_train:
                    print("   ⚠️  Leaf tip is in TRAIN set (should be in TEST!)")
                elif tip_in_test:
                    print("   ✅ Leaf tip is in TEST set (correct)")
        else:
            print(f"\n📊 Leaf Tip EPE: N/A (tip not found in one or both clouds)")
    else:
        print(f"\n📊 Leaf Tip EPE: N/A (is_leaf_tip attribute not found)")

    print("\n" + "=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate CD and EPE metrics between predicted and GT point clouds"
    )
    parser.add_argument(
        "--pred",
        type=str,
        default="pred.ply",
        help="Path to predicted PLY file",
    )
    parser.add_argument(
        "--gt",
        type=str,
        default="gt.ply",
        help="Path to ground truth PLY file",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    args.gt = Path(
        "datasets/Pheno4D/processed/v2/plant_sequences/maize/maize_control_plant2/sparse_pc/10_maize_control_plant2_D00_sparse.ply"
    )
    args.pred = Path(
        "results/pheno4d/ode_solve/maize/maize_control_plant2/20251111_052644_maize_control_plant2_P4D_full_l2_0.05_elas_0.05_prob_0.01_start_id_0_gap_9_stride_1/manifold/epoch_15000/10_maize_control_plant2_D00_sparse.ply"
    )
    main(args)
