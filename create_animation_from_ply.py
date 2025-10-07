import argparse
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def build_args():
    arg_parser = argparse.ArgumentParser(
        description="Create animation video from saved PLY files"
    )
    arg_parser.add_argument(
        "--ply_folder",
        "-p",
        dest="ply_folder",
        required=True,
        help="Path to folder containing PLY files (e.g., results/experiment/manifold)",
    )
    arg_parser.add_argument(
        "--fps",
        dest="fps",
        type=int,
        default=2,
        help="Frames per second for output video (default: 2)",
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
        help="Reverse file order (default: False)",
    )

    args = arg_parser.parse_args()
    return args


def render_mesh_frame(mesh, camera_angle, resolution=(1920, 1080), vis=None):
    """Render a single frame of the mesh from a given camera angle using Open3D."""
    # Convert trimesh to open3d
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

    # Use original mesh colors if available
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        colors = mesh.visual.vertex_colors[:, :3] / 255.0
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    elif hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None:
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
        # Default green color
        colors = np.tile([0.5, 0.8, 0.5], (len(mesh.vertices), 1))
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
    camera_distance = extent * 4.5

    # Camera position - looking from above (top-down view)
    cam_x = centroid[0]
    cam_y = centroid[1]
    cam_z = centroid[2] + camera_distance  # Above the object

    # Set view control
    ctr = vis.get_view_control()
    ctr.set_lookat(centroid)
    # Look straight down
    ctr.set_front([0, 0, -1])  # Looking down along Z axis
    ctr.set_up([0, 1, 0])  # Y axis is up in the view
    ctr.set_zoom(0.7)

    # Update and capture
    vis.poll_events()
    vis.update_renderer()

    # Capture image
    image = vis.capture_screen_float_buffer(do_render=True)
    image = np.asarray(image)

    return image, vis


def create_animation_from_ply(args):
    """Create animation video from saved PLY files."""

    folder = Path(args.ply_folder)

    # Get all PLY files sorted by name, excluding normalized_tempt
    ply_files = sorted(folder.glob("*.ply"))
    ply_files = [f for f in ply_files if "normalized_tempt" not in f.name]

    if len(ply_files) == 0:
        print(f"No PLY files found in {args.ply_folder}")
        return

    print(f"Found {len(ply_files)} PLY files:")
    for i, f in enumerate(ply_files):
        print(f"  {i}: {f.name}")

    # Optionally reverse order
    if args.reverse:
        ply_files = list(reversed(ply_files))
        print("\nREVERSED file order")

    # Create output directory for frames
    frames_dir = folder / "animation_frames_from_ply"
    if frames_dir.exists():
        import shutil

        shutil.rmtree(frames_dir)
    frames_dir.mkdir(exist_ok=True)

    # Calculate rotation angles
    rotation_angles = np.linspace(0, 2 * np.pi * args.num_rotations, len(ply_files))

    print(f"\nGenerating {len(ply_files)} frames...")
    vis = None

    for frame_idx, (ply_file, angle) in enumerate(
        tqdm(zip(ply_files, rotation_angles), total=len(ply_files))
    ):
        # Load mesh
        mesh = trimesh.load(str(ply_file))

        # Print mesh statistics
        bbox_size = mesh.bounds[1] - mesh.bounds[0]
        bbox_volume = np.prod(bbox_size)
        num_vertices = len(mesh.vertices)
        print(f"\nFrame {frame_idx} ({ply_file.name}):")
        print(f"  Bounding box size: {bbox_size}")
        print(f"  Bounding box volume: {bbox_volume:.2f}")
        print(f"  Number of vertices: {num_vertices}")

        # Render frame
        frame_img, vis = render_mesh_frame(
            mesh, angle, resolution=tuple(args.resolution), vis=vis
        )

        # Add text overlay with frame info
        frame_pil = Image.fromarray((frame_img * 255).astype(np.uint8))
        draw = ImageDraw.Draw(frame_pil)

        # Add text with frame number and filename
        text = f"Frame {frame_idx} | {ply_file.name}"
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
        frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
        frame_pil.save(frame_path)

    # Close visualizer
    if vis is not None:
        vis.destroy_window()

    # Create video using ffmpeg
    video_path = folder / "animation_from_ply.mp4"
    print(f"\nCreating video at {video_path}...")

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(video_path),
    ]

    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Video created successfully: {video_path}")
    print("\nKeeping frames in:", frames_dir)

    return video_path


if __name__ == "__main__":
    args = build_args()
    video_path = create_animation_from_ply(args)
