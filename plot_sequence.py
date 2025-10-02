import colorsys
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def visualize_ply_files_in_folder(
    folder_path: str,
    spacing: float = 30.0,
    show_axes: bool = False,
    window_name: str = "PLY Files Sequence",
):
    """
    Visualize all PLY files in a folder, spacing them along the Y axis

    Args:
        folder_path: Path to folder containing PLY files
        spacing: Distance between point clouds in the visualization
        show_axes: Whether to show principal axes for each point cloud
        window_name: Name for the visualization window
    """
    folder = Path(folder_path)

    # Get all PLY files sorted by name
    ply_files = sorted(folder.glob("*.ply"))
    ply_files = [file for file in ply_files if "normalized_tempt" not in file.name]

    if len(ply_files) == 0:
        print(f"No PLY files found in {folder_path}")
        return

    print(f"Found {len(ply_files)} PLY files")
    print("File order (left to right with increasing Y):")

    # Generate colors for each file using HSV colorspace
    colors = [
        np.array(colorsys.hsv_to_rgb(i / len(ply_files), 0.8, 0.9))
        for i in range(len(ply_files))
    ]

    # Color names based on HSV hue
    color_names = [
        "RED",
        "ORANGE",
        "YELLOW",
        "GREEN",
        "CYAN",
        "BLUE",
        "PURPLE",
        "MAGENTA",
    ]

    print("\nCOLOR LEGEND:")
    for i, f in enumerate(ply_files):
        rgb = (colors[i] * 255).astype(int)
        # Get approximate color name
        color_idx = int((i / len(ply_files)) * len(color_names)) % len(color_names)
        color_name = color_names[color_idx]
        print(
            f"  Position {i} (Y={i*spacing}): {f.name} -> {color_name} RGB({rgb[0]}, {rgb[1]}, {rgb[2]})"
        )

    geometries = []

    # Process each PLY file
    for i, ply_file in enumerate(ply_files):
        print(f"Loading {ply_file.name}...")

        # Read the mesh or point cloud
        try:
            mesh = o3d.io.read_triangle_mesh(str(ply_file))
            if len(mesh.vertices) == 0:
                # Try as point cloud
                pcd = o3d.io.read_point_cloud(str(ply_file))
                points = np.asarray(pcd.points)
            else:
                # Convert mesh to point cloud for visualization
                pcd = mesh.sample_points_uniformly(number_of_points=10000)
                points = np.asarray(pcd.points)
        except:
            print(f"Warning: {ply_file.name} could not be loaded, skipping")
            continue

        if len(points) == 0:
            print(f"Warning: {ply_file.name} is empty, skipping")
            continue

        # Translate each point cloud along Y axis for spacing
        translated_points = points.copy()
        translated_points[:, 1] += i * spacing

        pcd_new = o3d.geometry.PointCloud()
        pcd_new.points = o3d.utility.Vector3dVector(translated_points)

        # Color the point cloud
        point_colors = np.tile(colors[i], (len(points), 1))
        pcd_new.colors = o3d.utility.Vector3dVector(point_colors)

        geometries.append(pcd_new)

        # Add text label (as a small coordinate frame at the top of each scan)
        text_pos = translated_points.mean(axis=0)
        text_pos[2] = translated_points[:, 2].max() + 5  # Above the scan

        # Create a small coordinate frame as a marker
        if show_axes:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=5.0, origin=text_pos
            )
            geometries.append(coord_frame)

    # Visualize all geometries
    print(f"\nVisualizing {len(geometries)} objects...")
    print("NOTE: Meshes are spaced along Y-axis from left to right")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"{window_name} - {folder.name}",
    )


sys.argv

# Example usage
visualize_ply_files_in_folder(
    "results/tomato2_control_plant2_leaf1_P4D_full_l2_0.05_elas_0.05_prob_0.01_start_id_0_gap_8_stride_1/manifold",
    spacing=5,
)
