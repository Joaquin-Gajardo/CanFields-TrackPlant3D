import torch
import numpy as np
from pytorch3d.ops.knn import knn_points
from typing import Dict, Optional, Tuple


class PlantMetrics:
    """
    Metrics for evaluating 4D plant reconstruction quality.

    Includes:
    - Chamfer Distance (CD): Measures overall point cloud similarity
    - End Point Error (EPE): Measures accuracy of leaf tip predictions
    """

    def __init__(self):
        pass

    @staticmethod
    def chamfer_distance_numpy_one_way(src: np.ndarray, tgt: np.ndarray) -> float:
        """
        Compute one-way Chamfer Distance from src to tgt using numpy.

        Args:
            src: Source point cloud of shape (N, 3)
            tgt: Target point cloud of shape (M, 3)

        Returns:
            Mean of minimum squared distances
        """
        squared_distances = ((src[:, None, :] - tgt[None, :, :]) ** 2).sum(axis=2)
        min_distances = squared_distances.min(axis=1)
        return min_distances.mean()

    @staticmethod
    def chamfer_distance_numpy_symmetric(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Compute symmetric Chamfer Distance using numpy (sum of both directions).

        Args:
            pred: Predicted point cloud of shape (N, 3)
            gt: Ground truth point cloud of shape (M, 3)

        Returns:
            Sum of bidirectional Chamfer distances
        """
        cd_pred_to_gt = PlantMetrics.chamfer_distance_numpy_one_way(pred, gt)
        cd_gt_to_pred = PlantMetrics.chamfer_distance_numpy_one_way(gt, pred)
        return cd_pred_to_gt + cd_gt_to_pred

    @staticmethod
    def chamfer_distance(
        pred_points: torch.Tensor,
        gt_points: torch.Tensor,
        reduction: str = "mean",
        bidirectional: bool = True,
    ) -> torch.Tensor:
        """
        Compute Chamfer Distance between predicted and ground truth point clouds.

        Args:
            pred_points: Predicted point cloud of shape (N, 3) or (B, N, 3)
            gt_points: Ground truth point cloud of shape (M, 3) or (B, M, 3)
            reduction: "mean", "sum", or "none"
            bidirectional: If True, compute symmetric Chamfer distance

        Returns:
            Chamfer distance value(s)
        """
        # Convert to numpy for computation
        if pred_points.dim() == 2:
            pred_np = pred_points.cpu().numpy()
            gt_np = gt_points.cpu().numpy()

            if bidirectional:
                cd = PlantMetrics.chamfer_distance_numpy_symmetric(pred_np, gt_np)
            else:
                cd = PlantMetrics.chamfer_distance_numpy_one_way(pred_np, gt_np)

            return torch.tensor(cd, dtype=torch.float32)
        else:
            # Batch mode - compute for each batch element
            batch_size = pred_points.shape[0]
            cds = []
            for i in range(batch_size):
                pred_np = pred_points[i].cpu().numpy()
                gt_np = gt_points[i].cpu().numpy()

                if bidirectional:
                    cd = PlantMetrics.chamfer_distance_numpy_symmetric(pred_np, gt_np)
                else:
                    cd = PlantMetrics.chamfer_distance_numpy_one_way(pred_np, gt_np)

                cds.append(cd)

            return torch.tensor(cds, dtype=torch.float32)

    @staticmethod
    def endpoint_error(
        pred_tips: torch.Tensor,
        gt_tips: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute End Point Error (EPE) for leaf tips.

        Args:
            pred_tips: Predicted leaf tip positions of shape (K, 3) or (B, K, 3)
            gt_tips: Ground truth leaf tip positions of shape (K, 3) or (B, K, 3)
            reduction: "mean", "sum", or "none"

        Returns:
            EPE value(s) - Euclidean distance between corresponding tips
        """
        # Compute Euclidean distance
        distances = torch.norm(pred_tips - gt_tips, dim=-1)

        if reduction == "mean":
            return distances.mean()
        elif reduction == "sum":
            return distances.sum()
        else:  # "none"
            return distances


    @staticmethod
    def compute_metrics_per_timestep(
        pred_points_list: list,
        gt_points_list: list,
        pred_tips_list: Optional[list] = None,
        gt_tips_list: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute metrics for all timesteps in a sequence.

        Args:
            pred_points_list: List of predicted point clouds, each (N, 3)
            gt_points_list: List of ground truth point clouds, each (M, 3)
            pred_tips_list: Optional list of predicted leaf tips, each (K, 3)
            gt_tips_list: Optional list of ground truth leaf tips, each (K, 3)

        Returns:
            Dictionary containing:
                - "chamfer_distances": Tensor of shape (T,) with CD for each timestep
                - "epe": Tensor of shape (T,) with EPE for each timestep (if tips provided)
                - "mean_chamfer": Scalar mean CD across all timesteps
                - "mean_epe": Scalar mean EPE across all timesteps (if tips provided)
        """
        assert len(pred_points_list) == len(gt_points_list), \
            "Number of predicted and GT point clouds must match"

        num_timesteps = len(pred_points_list)
        chamfer_distances = []

        for i in range(num_timesteps):
            cd = PlantMetrics.chamfer_distance(
                pred_points_list[i],
                gt_points_list[i],
                reduction="mean",
                bidirectional=True
            )
            chamfer_distances.append(cd.squeeze())

        chamfer_distances = torch.stack(chamfer_distances)

        metrics = {
            "chamfer_distances": chamfer_distances,
            "mean_chamfer": chamfer_distances.mean(),
            "std_chamfer": chamfer_distances.std(),
        }

        # Compute EPE if tips are provided
        if pred_tips_list is not None and gt_tips_list is not None:
            assert len(pred_tips_list) == len(gt_tips_list), \
                "Number of predicted and GT tips must match"

            epe_values = []
            # Iterate over the tips list length (may be less than num_timesteps)
            for i in range(len(pred_tips_list)):
                epe = PlantMetrics.endpoint_error(
                    pred_tips_list[i],
                    gt_tips_list[i],
                    reduction="mean"
                )
                epe_values.append(epe.squeeze())

            epe_values = torch.stack(epe_values)
            metrics["epe"] = epe_values
            metrics["mean_epe"] = epe_values.mean()
            metrics["std_epe"] = epe_values.std()

        return metrics


def chamfer_distance_simple(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    batch_reduction: str = "mean",
    point_reduction: str = "mean",
) -> torch.Tensor:
    """
    Simple wrapper for Chamfer Distance computation.
    Compatible with existing codebase style.

    Args:
        pred_points: Predicted points (N, 3) or (B, N, 3)
        gt_points: Ground truth points (M, 3) or (B, M, 3)
        batch_reduction: "mean", "sum", or None
        point_reduction: "mean" or "sum"

    Returns:
        Chamfer distance
    """
    metrics = PlantMetrics()
    return metrics.chamfer_distance(
        pred_points, gt_points, reduction=point_reduction, bidirectional=True
    )
