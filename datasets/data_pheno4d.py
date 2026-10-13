import json
import os

import numpy as np
import open3d as o3d
import point_cloud_utils as pcu
import torch


class DeformSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        filename,
        start_idx=0,
        gap=None,
        train_step=1,
        use_dense=False,
        split_file=None,
        use_train_points=True,
        interp_stride=None,
    ):
        """
        Pheno4D dataset loader for plant sequences.

        Args:
            filename: Path to specific plant sequence (e.g., "datasets/Pheno4D/maize/maize_control_plant2_leaf2")
            start_idx: Starting frame index (default: 0)
            gap: Number of consecutive frames to include (None = use all available frames)
            train_step: Sampling step for temporal sequence (1 = all frames, 2 = every other frame, etc.)
                       Note: ignored if interp_stride is set
            use_dense: Whether to use dense_pc (True) or sparse_pc (False)
            split_file: Path to JSON file with train/test point splits (optional)
            use_train_points: If split_file provided, use train points (True) or test points (False)
            interp_stride: If set, enables interpolation mode where all timepoints are loaded but only
                          every Nth timepoint (by position, not day number) is used for training.
                          E.g., interp_stride=2 means position 0,2,4,... are train, position 1,3,5,... are test.
                          Overrides train_step when set.
        """
        pc_folder = "dense_pc" if use_dense else "sparse_pc"
        pc_path = os.path.join(filename, pc_folder)

        if not os.path.exists(pc_path):
            raise ValueError(f"Path does not exist: {pc_path}")

        # Get all .ply files and sort them by timestamp
        sub_files = [f for f in os.listdir(pc_path) if f.endswith(".ply")]

        # Extract timestamp from filename and sort
        def extract_timestamp(fname):
            # Format: "12_maize_control_plant2_D02_leaf2_dense.ply" or "361_tomato2_control_plant2_D00_leaf1.ply"
            # Extract the day number (D02 -> 2)
            parts = fname.split("_")
            for part in parts:
                if part.startswith("D") and len(part) >= 2:
                    try:
                        # Handle formats like D00, D01, D02, etc.
                        day_num = part[1:]
                        if day_num.isdigit():
                            return int(day_num)
                    except ValueError:
                        continue
            # Fallback: extract first number from filename
            try:
                return int(fname.split("_")[0])
            except ValueError:
                return 0

        sub_files = sorted(sub_files, key=extract_timestamp)
        print("sub_files: ", sub_files)
        total_lens = len(sub_files)

        if total_lens == 0:
            raise ValueError(f"No .ply files found in {pc_path}")

        # Store interpolation stride setting
        self.interp_stride = interp_stride

        # Apply start_idx and gap filtering (similar to data_wmw.py)
        # In interpolation mode, load ALL timepoints but mark which are for training
        if gap is not None:
            end_idx = min(start_idx + gap, total_lens)
            if interp_stride is not None:
                # Interpolation mode: load all timepoints in the gap
                sub_files = sub_files[start_idx:end_idx]
            else:
                # Normal mode: use train_step to filter
                sub_files = sub_files[start_idx:end_idx:train_step]
        else:
            if interp_stride is not None:
                # Interpolation mode: load all timepoints
                sub_files = sub_files[start_idx:]
            else:
                # Normal mode: use train_step to filter
                sub_files = sub_files[start_idx::train_step]

        if interp_stride is not None:
            print(
                f"Interpolation mode: loaded all timepoints (start_idx={start_idx}, gap={gap}, interp_stride={interp_stride})"
            )
        else:
            print(
                f"Filtered sub_files (start_idx={start_idx}, gap={gap}, train_step={train_step}): ",
                sub_files,
            )

        if len(sub_files) == 0:
            raise ValueError(
                f"No files remaining after filtering with start_idx={start_idx}, gap={gap}, train_step={train_step}"
            )

        self.train_xx = []
        self.train_nn = []
        self.normalized_times = []
        self.is_train_timestep = []  # Track which timesteps are for training (interpolation mode)

        # Load point splits if provided
        self.split_file = split_file
        self.use_train_points = use_train_points
        self.point_splits = None
        if split_file and os.path.exists(split_file):
            with open(split_file, "r") as f:
                self.point_splits = json.load(f)
            print(f"Loaded point splits from {split_file}")
            print(
                f"Using {'train' if use_train_points else 'test'} points for this dataset"
            )
        elif split_file:
            print(f"Warning: split_file specified but not found: {split_file}")

        min_norm_normal = 1e-5
        train_lens = 0

        # Calculate time normalization based on actual timestamps of filtered files
        timestamps = [extract_timestamp(f) for f in sub_files]
        min_time = min(timestamps)
        max_time = max(timestamps)
        time_range = max_time - min_time if max_time > min_time else 1

        for i, sub_file in enumerate(sub_files):
            timestamp = timestamps[i]

            # Normalize time to [0, 1]
            normalized_time = (timestamp - min_time) / time_range
            self.normalized_times.append(torch.tensor(normalized_time))

            # Track if this timestep is for training (interpolation mode)
            if self.interp_stride is not None:
                # In interpolation mode: position 0, stride, 2*stride, ... are training
                is_train = (i % self.interp_stride) == 0
                self.is_train_timestep.append(is_train)
            else:
                # In normal mode: all loaded timesteps are for training
                self.is_train_timestep.append(True)

            full_name = os.path.join(pc_path, sub_file)

            try:
                # Try loading with Open3D tensor to get scalar fields
                pcd_tensor = o3d.t.io.read_point_cloud(full_name)
                v = pcd_tensor.point.positions.numpy().astype(np.float32)

                # Read leaf tips if available
                leaf_tip_mask = None
                if "is_leaf_tip" in pcd_tensor.point:
                    leaf_tip_mask = (
                        pcd_tensor.point.is_leaf_tip.numpy().flatten().astype(bool)
                    )

                # Read normals if available
                n = None
                if "normals" in pcd_tensor.point:
                    n = pcd_tensor.point.normals.numpy().astype(np.float32)

                # # Old with PCU
                # v, _, n = pcu.load_mesh_vfn(full_name, dtype=np.float32)
                # leaf_tip_mask = None

                # Drop duplicates and points with near-zero normals as a mask, keeping the
                # file's point order: the point split indices refer to that order.
                _, idx, _ = pcu.deduplicate_point_cloud(v, 1e-15, return_index=True)
                keep = np.zeros(len(v), dtype=bool)
                keep[idx] = True

                if n is not None:
                    keep &= np.linalg.norm(n, axis=-1) > min_norm_normal
                else:
                    # No normals available, estimate them with Open3D on the kept points
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(v[keep].astype(np.float64))
                    pcd.estimate_normals()
                    pcd.orient_normals_consistent_tangent_plane(k=30)
                    pcd.orient_normals_to_align_with_direction()
                    n = np.zeros_like(v, dtype=np.float32)
                    n[keep] = np.asarray(pcd.normals)

                selected = np.flatnonzero(keep)
                # Fail instead of skipping: a skipped frame would shift every later frame's
                # time and train flag, which are appended above
                if len(selected) == 0:
                    raise ValueError(f"No valid points in {sub_file}")

                # Apply point splits if provided
                if self.point_splits is not None:
                    # Get the split indices for this timepoint
                    timepoint_data = self.point_splits["timepoints"][i]
                    if self.use_train_points:
                        split_indices = np.array(
                            timepoint_data["train_indices"], dtype=np.int64
                        )
                    else:
                        split_indices = np.array(
                            timepoint_data["test_indices"], dtype=np.int64
                        )

                    # Ensure indices are within bounds, then drop the removed points
                    split_indices = split_indices[split_indices < len(v)]
                    selected = split_indices[keep[split_indices]]

                    if len(selected) == 0:
                        raise ValueError(
                            f"No valid split indices for {sub_file} timepoint {i}"
                        )

                    print(
                        f"  Applied point split to timepoint {i}: {len(selected)} points"
                    )

                x = v[selected].astype(np.float32)
                n = n[selected].astype(np.float32)
                n /= np.linalg.norm(n, axis=-1, keepdims=True)
                if leaf_tip_mask is not None:
                    leaf_tip_mask = leaf_tip_mask[selected]

                self.train_xx.append(torch.tensor(x))
                self.train_nn.append(torch.tensor(n))

                # Store leaf tip indices if available
                if leaf_tip_mask is not None:
                    leaf_tip_idxs = torch.tensor(
                        np.where(leaf_tip_mask)[0], dtype=torch.long
                    )
                else:
                    leaf_tip_idxs = torch.tensor([], dtype=torch.long)

                # Store leaf tips - will be used for EPE evaluation
                if not hasattr(self, "leaf_tips"):
                    self.leaf_tips = []
                self.leaf_tips.append(leaf_tip_idxs)

                train_lens += 1

            except Exception as e:
                raise ValueError(f"Error loading {sub_file}: {e}") from e

        self.lens = train_lens
        self.sub_files = sub_files

        if self.lens == 0:
            raise ValueError("No valid point clouds loaded")

        # Compute normalization parameters
        big_pc = torch.cat(self.train_xx, dim=0)
        self.mean = big_pc.mean(0, keepdim=True)
        self.scale = 1.0 / torch.max(torch.abs(big_pc - self.mean))
        big_pc = (big_pc - self.mean) * self.scale
        self.bb_min = big_pc.min(0, keepdim=True)[0]
        self.bb_max = big_pc.max(0, keepdim=True)[0]
        del big_pc

        # Apply normalization to all point clouds
        for idx in range(int(self.lens)):
            self.train_xx[idx] = (self.train_xx[idx] - self.mean) * self.scale

        # Print summary of train/test split in interpolation mode
        if self.interp_stride is not None:
            train_count = sum(self.is_train_timestep)
            test_count = len(self.is_train_timestep) - train_count
            print(f"\nInterpolation mode summary:")
            print(f"  Total timesteps: {len(self.is_train_timestep)}")
            print(f"  Training timesteps: {train_count}")
            print(f"  Testing timesteps: {test_count}")
            print(f"  Training indices: {[i for i, is_train in enumerate(self.is_train_timestep) if is_train]}")
            print(f"  Testing indices: {[i for i, is_train in enumerate(self.is_train_timestep) if not is_train]}")

    def __len__(self):
        return self.lens

    def evaluate_train(self, idx):
        pts = self.train_xx[idx]
        return pts

    def get_leaf_tips(self, idx):
        """Get leaf tip indices for a specific timestep."""
        if hasattr(self, "leaf_tips") and idx < len(self.leaf_tips):
            return self.leaf_tips[idx]
        return torch.tensor([], dtype=torch.long)

    def is_training_timestep(self, idx):
        """Check if a timestep is used for training (relevant in interpolation mode)."""
        if idx < len(self.is_train_timestep):
            return self.is_train_timestep[idx]
        return True  # Default to training if not set

    def __getitem__(self, idx):
        pts = self.train_xx[idx]
        nms = self.train_nn[idx]
        normalized_time = self.normalized_times[idx]
        is_train = self.is_train_timestep[idx] if idx < len(self.is_train_timestep) else True
        return {"pts": pts, "nms": nms, "normed_time": normalized_time, "is_train": is_train}


if __name__ == "__main__":
    # Test the dataset loader
    path = "datasets/Pheno4D/maize/maize_control_plant2_leaf2"
    if os.path.exists(path):
        # Test with gap parameter
        dataset = DeformSamples(path, start_idx=0, gap=9, train_step=1, use_dense=False)
        print("lens: ", len(dataset))
        print("normalized_times: ", dataset.normalized_times)
        for i in range(len(dataset)):
            dps = dataset[i]
            print(
                f"Frame {i}: pts.shape: {dps['pts'].shape}, normed_time: {dps['normed_time']}"
            )
    else:
        print(f"Test path {path} does not exist")
