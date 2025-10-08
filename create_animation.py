import argparse
import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import trimesh
from matplotlib import cm
from tqdm import tqdm

from eval_sphere import (
    get_available_checkpoints,
    load_model_latents,
    select_epochs_to_evaluate,
)
from utils import setup_seed


def build_args():
    arg_parser = argparse.ArgumentParser(
        description="Create animation video of temporal reconstruction"
    )
    arg_parser.add_argument(
        "--specifications",
        default="configs/specs.json",
        dest="specifications",
        required=False,
        help="The experiment directory with specifications",
    )
    arg_parser.add_argument(
        "--base_results_dir",
        default="results",
        dest="experiment_directory",
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
        required=False,
        help="Object name",
    )
    arg_parser.add_argument(
        "--experiment_name",
        "-en",
        dest="experiment_name",
        required=False,
        help="Name of the specific experiment",
    )
    arg_parser.add_argument(
        "--dataset_name",
        "-d",
        dest="dataset_name",
        required=True,
        help="Dataset name: 'animals','medical','defaust','pheno4d'",
    )
    arg_parser.add_argument(
        "--epoch",
        "-e",
        dest="epoch",
        required=True,
        help="Checkpoint selection: 'all' for all checkpoints, 'last' for most recent, or specific epoch number (e.g., 15000)",
    )
    arg_parser.add_argument(
        "--step",
        "-s",
        dest="step",
        default="2",
        required=False,
        help="Step for evaluation",
    )
    arg_parser.add_argument(
        "--multi_step",
        "-m",
        dest="multi_step",
        required=True,
        help="Multi step parameter",
    )
    arg_parser.add_argument(
        "--num_frames",
        "-nf",
        dest="num_frames",
        type=int,
        default=100,
        help="Number of frames for animation (default: 100)",
    )
    arg_parser.add_argument(
        "--fps",
        dest="fps",
        type=int,
        default=30,
        help="Frames per second for output video (default: 30)",
    )
    arg_parser.add_argument(
        "--num_rotations",
        "-nr",
        dest="num_rotations",
        type=float,
        default=0.0,
        help="Number of full rotations around object (default: 0.0 for fixed camera)",
    )
    arg_parser.add_argument(
        "--resolution",
        "-res",
        dest="resolution",
        type=int,
        nargs=2,
        default=[1920, 1080],
        help="Video resolution width height (default: 1920 1080)",
    )
    arg_parser.add_argument(
        "--reverse",
        "-r",
        dest="reverse",
        action="store_true",
        help="Reverse time direction (default: False, goes from early to late)",
    )

    args = arg_parser.parse_args()
    return args


def render_mesh_frame(
    mesh,
    camera_angle,
    actual_scan_times,
    current_time,
    resolution=(1920, 1080),
    highlight_scans=True,
    use_mesh_colors=False,
    vis=None,
):
    """
    Render a single frame of the mesh from a given camera angle using Open3D.

    Args:
        mesh: trimesh object
        camera_angle: rotation angle in radians
        actual_scan_times: list of normalized times where actual scans exist
        current_time: current normalized time being rendered
        resolution: tuple of (width, height)
        highlight_scans: whether to highlight if this is an actual scan time
        vis: Open3D visualizer (reused for efficiency)
    """
    # Convert trimesh to open3d
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

    # Determine if this is an actual scan time
    is_scan_time = False
    if highlight_scans and actual_scan_times is not None:
        tolerance = 0.02
        is_scan_time = any(abs(current_time - t) < tolerance for t in actual_scan_times)

    # Set vertex colors
    if is_scan_time:
        # Gold/yellow color for actual scan times
        colors = np.tile([1.0, 0.843, 0.0], (len(mesh.vertices), 1))
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    else:
        # Use original mesh colors if available
        if (
            hasattr(mesh.visual, "vertex_colors")
            and mesh.visual.vertex_colors is not None
            and use_mesh_colors
        ):
            colors = mesh.visual.vertex_colors[:, :3] / 255.0
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        elif (
            hasattr(mesh.visual, "face_colors")
            and mesh.visual.face_colors is not None
            and use_mesh_colors
        ):
            # Convert face colors to vertex colors
            face_colors = mesh.visual.face_colors[:, :3] / 255.0
            vertex_colors = np.zeros((len(mesh.vertices), 3))
            counts = np.zeros(len(mesh.vertices))
            for i, face in enumerate(mesh.faces):
                for v in face:
                    vertex_colors[v] += face_colors[i]
                    counts[v] += 1
            vertex_colors /= np.maximum(counts[:, None], 1)
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        else:
            # Default gray color
            colors = np.tile([0.7, 0.7, 0.7], (len(mesh.vertices), 1))
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Compute normals for better lighting
    o3d_mesh.compute_vertex_normals()

    # Create or reuse visualizer
    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=resolution[0], height=resolution[1], visible=False)
        vis.add_geometry(o3d_mesh)

        # Set render options
        opt = vis.get_render_option()
        opt.light_on = True
        opt.mesh_show_back_face = False
    else:
        vis.clear_geometries()
        vis.add_geometry(o3d_mesh)

    # Calculate camera parameters
    bounds = mesh.bounds
    centroid = mesh.centroid
    extent = np.linalg.norm(bounds[1] - bounds[0])
    camera_distance = extent * 4.5  # Increased to 4.5 to see full object

    # Camera position rotating around the object (in horizontal plane, looking sideways)
    cam_x = centroid[0] + camera_distance * np.cos(camera_angle)
    cam_z = centroid[2] + camera_distance * np.sin(camera_angle)
    cam_y = centroid[1]  # At same height as centroid for sideways view

    # Set view control
    ctr = vis.get_view_control()
    ctr.set_lookat(centroid)
    ctr.set_front([centroid[0] - cam_x, centroid[1] - cam_y, centroid[2] - cam_z])
    ctr.set_up([0, 0, 1])  # Z-axis is up (vertical objects appear horizontal)
    ctr.set_zoom(0.5)

    # Update and capture
    vis.poll_events()
    vis.update_renderer()

    # Capture image
    image = vis.capture_screen_float_buffer(do_render=True)
    image = np.asarray(image)

    return image, vis


def create_animation(args, epoch_num):
    """Create animation video from temporal reconstruction for a specific epoch."""

    # Load model and data
    print(f"Loading model and data for epoch {epoch_num}...")
    if args.methods != "ode_solve" or "woAdaptor" in args.object_name:
        (
            args_loaded,
            sdf_dataset,
            decoder,
            start_end_vecs,
            start_idx,
            end_idx,
            export_path,
        ) = load_model_latents(args, epoch_num)
        ajust = None
        adjust_start_end_vecs = None
    else:
        (
            args_loaded,
            sdf_dataset,
            decoder,
            ajust,
            adjust_start_end_vecs,
            start_idx,
            end_idx,
            export_path,
        ) = load_model_latents(args, epoch_num)
        start_end_vecs = None

    # Update args with loaded values
    for key, value in vars(args_loaded).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # Check if template mesh exists
    templat_full_path = os.path.join(export_path, "normalized_tempt.ply")
    if not os.path.exists(templat_full_path):
        print(f"Template mesh not found at {templat_full_path}")
        print("Please run: python eval_sphere.py -s 1 [other args] first")
        sys.exit(1)

    # Load template mesh
    print(f"Loading template mesh from {templat_full_path}")
    temp_mesh = trimesh.load(templat_full_path)
    temp_v = np.asarray(temp_mesh.vertices, dtype=np.float32)
    temp_f = np.asarray(temp_mesh.faces)
    temp_v = torch.tensor(temp_v).cuda()

    # Get time range
    start_time = sdf_dataset[start_idx]["normed_time"].cuda()
    end_time = sdf_dataset[end_idx]["normed_time"].cuda()
    decoder.warper.split_time(start_time, end_time)
    gt_time = 0.5 * (start_time + end_time)

    # Get actual scan times for highlighting
    actual_scan_times = [
        float(sdf_dataset[i]["normed_time"]) for i in range(len(sdf_dataset))
    ]
    print(f"Actual scan times: {actual_scan_times}")
    print(f"Start time: {start_time.item()}, End time: {end_time.item()}")
    print(f"Dataset sub_files order: {sdf_dataset.sub_files}")
    print("\nFile to time mapping:")
    for i, (f, t) in enumerate(zip(sdf_dataset.sub_files, actual_scan_times)):
        print(f"  {i}: {f} -> time {t:.4f}")

    # DEBUG MODE: Use only actual scan times (8 frames) instead of dense sampling
    if args.num_frames <= len(sdf_dataset):
        print(f"DEBUG MODE: Using {len(sdf_dataset)} actual scan times only")
        dense_times = sdf_dataset.normalized_times
        args.num_frames = len(dense_times)  # Update frame count
        # Convert list to tensor if needed
        if isinstance(dense_times, list):
            dense_times = torch.stack(dense_times).cuda()
    else:
        # Create dense time samples for animation (interpolating between actual scans)
        dense_times = torch.linspace(start_time, end_time, args.num_frames).cuda()
        print(f"Animation will interpolate across {args.num_frames} time steps")

    print(f"Times to render: {[float(t) for t in dense_times]}")

    # Optionally reverse the time direction
    if args.reverse:
        dense_times = torch.flip(dense_times, [0])
        print(f"REVERSED time order: {[float(t) for t in dense_times]}")

    # Dataset normalization
    d_s = sdf_dataset.scale
    d_m = sdf_dataset.mean

    # Create output directory for animations under experiment_directory
    reverse_suffix = "_reversed" if args.reverse else ""
    animations_dir = os.path.join(
        export_path,
        "..",
        "animations",
        f"epoch_{epoch_num}_{args.step}steps_{args.fps}fps{reverse_suffix}",
    )
    frames_dir = os.path.join(animations_dir, "frames")

    if os.path.exists(animations_dir):
        # Clear old animation
        import shutil

        shutil.rmtree(animations_dir)
    os.makedirs(frames_dir, exist_ok=True)

    # Calculate rotation angles
    rotation_angles = np.linspace(0, 2 * np.pi * args.num_rotations, args.num_frames)

    print(f"Generating {args.num_frames} frames...")
    print(f"gt_time (canonical): {gt_time.item()}")
    num_chunks = 5
    vis = None  # Reuse visualizer for efficiency

    for frame_idx, (back_time, angle) in enumerate(
        tqdm(zip(dense_times, rotation_angles), total=args.num_frames)
    ):
        print(
            f"Frame {frame_idx}: warping from gt_time={gt_time.item():.4f} to back_time={back_time.item():.4f}"
        )
        # Warp vertices to this time
        chunks = torch.chunk(temp_v, num_chunks, dim=0)
        v_final = []
        for chunk in chunks:
            chunk_final = decoder.warper(chunk, gt_time, back_time)
            v_final.append(chunk_final)
        v_final = torch.cat(v_final, dim=0) / d_s.cuda() + d_m.cuda()

        # Create mesh
        warped_mesh = trimesh.Trimesh(
            vertices=v_final.detach().clone().cpu().numpy(), faces=temp_f
        )
        if hasattr(temp_mesh.visual, "face_colors"):
            warped_mesh.visual.face_colors = temp_mesh.visual.face_colors

        # Print mesh statistics
        bbox_size = warped_mesh.bounds[1] - warped_mesh.bounds[0]
        bbox_volume = np.prod(bbox_size)
        num_vertices = len(warped_mesh.vertices)
        print(f"  Bounding box size: {bbox_size}")
        print(f"  Bounding box volume: {bbox_volume:.2f}")
        print(f"  Number of vertices: {num_vertices}")

        # Render frame
        frame_img, vis = render_mesh_frame(
            warped_mesh,
            angle,
            actual_scan_times,
            float(back_time.cpu()),
            resolution=tuple(args.resolution),
            highlight_scans=True,
            vis=vis,
        )

        # Add text overlay with frame info
        from PIL import Image, ImageDraw, ImageFont

        frame_pil = Image.fromarray((frame_img * 255).astype(np.uint8))
        draw = ImageDraw.Draw(frame_pil)

        # Add text with frame number and time
        text = f"Frame {frame_idx} | Time: {float(back_time.cpu()):.4f}"
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40
            )
        except:
            font = ImageFont.load_default()

        # Draw text with black outline for visibility
        x, y = 20, 20
        for adj in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
            draw.text((x + adj[0], y + adj[1]), text, font=font, fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=(255, 255, 255))

        # Save frame
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
        frame_pil.save(frame_path)

    # Close visualizer
    if vis is not None:
        vis.destroy_window()

    # Create video using ffmpeg
    video_path = os.path.join(animations_dir, "animation.mp4")
    print(f"\nCreating video at {video_path}...")

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        os.path.join(frames_dir, "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        video_path,
    ]

    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Video created successfully: {video_path}")

    # Optionally clean up frames
    print("\nKeeping frames in:", frames_dir)
    print("Delete this folder if you don't need the individual frames.")

    return video_path


if __name__ == "__main__":
    setup_seed(3407)
    args = build_args()

    # Import necessary args parsing from eval_sphere
    import sys

    # First, we need to determine the experiment directory to get available checkpoints
    from eval_sphere import build_args as eval_build_args

    sys.argv = [sys.argv[0]]  # Reset argv for eval_sphere's build_args

    # Reconstruct command line args for eval_sphere compatibility
    cmd_args = [
        "--dataset_name",
        args.dataset_name,
        "--epoch",
        (
            args.epoch if args.epoch.lower() not in ["all", "last"] else "1"
        ),  # Dummy epoch for arg parsing
        "--step",
        args.step,
        "--multi_step",
        args.multi_step,
    ]

    if args.object_name:
        cmd_args.extend(["--object_name", args.object_name])
    if args.experiment_name:
        cmd_args.extend(["--experiment_name", args.experiment_name])
    if args.methods:
        cmd_args.extend(["--methods", args.methods])

    sys.argv.extend(cmd_args)

    # Get experiment directory to find available checkpoints
    eval_args = eval_build_args()

    # Build experiment directory path
    import json

    specs = json.load(open(eval_args.specifications))
    obj_name = eval_args.object_name.split("/")[-1] if eval_args.object_name else ""
    experiment_folder = eval_args.experiment_name.split("/")[-1]

    experiment_path = os.path.join(
        eval_args.base_results_dir,
        eval_args.dataset_name,
        obj_name,
        eval_args.methods,
        experiment_folder,
    )
    # Get list of epochs to evaluate
    epochs_to_evaluate = select_epochs_to_evaluate(args.epoch, experiment_path)

    print(
        f"Will create animations for {len(epochs_to_evaluate)} epoch(s): {epochs_to_evaluate}"
    )

    # Create animations for each epoch
    for epoch_num in epochs_to_evaluate:
        print(f"\n{'='*80}")
        print(f"Creating animation for epoch {epoch_num}")
        print(f"{'='*80}\n")

        # Update args with current epoch
        args.epoch = str(epoch_num)

        # Reset sys.argv for this epoch
        sys.argv = [sys.argv[0]]
        cmd_args = [
            "--dataset_name",
            args.dataset_name,
            "--epoch",
            str(epoch_num),
            "--step",
            args.step,
            "--multi_step",
            args.multi_step,
        ]

        if args.object_name:
            cmd_args.extend(["--object_name", args.object_name])
        if args.experiment_name:
            cmd_args.extend(["--experiment_name", args.experiment_name])
        if args.methods:
            cmd_args.extend(["--methods", args.methods])

        sys.argv.extend(cmd_args)

        video_path = create_animation(args, epoch_num)
        print(f"Completed animation for epoch {epoch_num}: {video_path}")

    print(f"\n{'='*80}")
    print(f"All animations completed! Processed {len(epochs_to_evaluate)} epoch(s)")
    print(f"{'='*80}")
