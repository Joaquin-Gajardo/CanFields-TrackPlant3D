#!/usr/bin/env python3
"""
Create videos for all plant sequences and optionally stitch them together.

This script:
1. Finds all sequences in the results directory
2. Creates animation videos for each sequence (if not already created)
3. Optionally stitches all videos together into a grid layout

Usage:
    python create_all_videos_and_stitch.py --epoch last --fps 2
    python create_all_videos_and_stitch.py --epoch 15000 --fps 5 --no-stitch
    python create_all_videos_and_stitch.py --epoch last --stitch-only
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm


def build_args():
    parser = argparse.ArgumentParser(
        description="Create videos for all sequences and stitch them together"
    )
    parser.add_argument(
        "--results_dir",
        default="results/pheno4d/ode_solve",
        help="Base results directory (default: results/pheno4d/ode_solve)",
    )
    parser.add_argument(
        "--epoch",
        "-e",
        default="last",
        help="Epoch to use: 'last' for most recent, 'all', or specific number (default: last)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second for videos (default: 2)",
    )
    parser.add_argument(
        "--step",
        default="2",
        help="Step parameter for create_animation.py (default: 2)",
    )
    parser.add_argument(
        "--multi_step",
        default="1",
        help="Multi step parameter for create_animation.py (default: 1)",
    )
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="Skip video creation, only stitch existing videos",
    )
    parser.add_argument(
        "--no-stitch",
        action="store_true",
        help="Skip stitching, only create individual videos",
    )
    parser.add_argument(
        "--stitch-only",
        action="store_true",
        help="Only stitch existing videos (same as --no-create)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing videos",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=3,
        help="Number of columns in the stitched grid (default: 3)",
    )
    parser.add_argument(
        "--crop",
        help="Only process specific crop (maize, sorghum, tobacco, tomato)",
    )
    parser.add_argument(
        "--output-name",
        default="all_plants_stitched.mp4",
        help="Name for the stitched output video (default: all_plants_stitched.mp4)",
    )
    parser.add_argument(
        "--no-interpolation",
        action="store_true",
        help="Use actual scan times only (no interpolation between frames)",
    )
    parser.add_argument(
        "--group-by-crop",
        action="store_true",
        help="Create one video per crop, then stitch crops into 2x2 grid",
    )

    args = parser.parse_args()

    # Handle stitch-only flag
    if args.stitch_only:
        args.no_create = True

    return args


def find_sequences(results_dir: str, crop_filter: str = None) -> List[Tuple[str, str, str]]:
    """
    Find all sequences in the results directory.

    Returns:
        List of tuples: (crop, sequence_name, full_path)
    """
    results_path = Path(results_dir)
    sequences = []

    if not results_path.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return sequences

    # Find all crop directories
    for crop_dir in sorted(results_path.iterdir()):
        if not crop_dir.is_dir():
            continue

        crop_name = crop_dir.name
        if crop_filter and crop_name != crop_filter:
            continue

        # Find all sequence directories in this crop
        for seq_dir in sorted(crop_dir.iterdir()):
            if not seq_dir.is_dir():
                continue

            sequences.append((crop_name, seq_dir.name, str(seq_dir)))

    return sequences


def find_latest_experiment(sequence_path: str) -> str:
    """Find the most recent experiment folder in a sequence directory."""
    seq_path = Path(sequence_path)
    if not seq_path.exists():
        return None

    # Get all experiment folders (should be timestamped)
    exp_folders = [d for d in seq_path.iterdir() if d.is_dir()]
    if not exp_folders:
        return None

    # Sort by modification time, get most recent
    exp_folders.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(exp_folders[0])


def find_video_for_sequence(
    sequence_path: str, epoch: str, fps: int, step: str
) -> str:
    """
    Find existing animation video for a sequence.

    Returns:
        Path to video file, or None if not found
    """
    exp_folder = find_latest_experiment(sequence_path)
    if not exp_folder:
        return None

    animations_dir = Path(exp_folder) / "animations"
    if not animations_dir.exists():
        return None

    # Look for videos matching the criteria
    # Pattern: epoch_{epoch}_{step}steps_{fps}fps/animation.mp4
    if epoch == "last":
        # Find all epoch folders and pick the latest
        epoch_dirs = sorted(animations_dir.glob("epoch_*"))
        if not epoch_dirs:
            return None
        # Filter by step and fps
        matching = [
            d
            for d in epoch_dirs
            if f"_{step}steps_{fps}fps" in d.name and not "reversed" in d.name
        ]
        if not matching:
            return None
        video_path = matching[-1] / "animation.mp4"
    else:
        # Look for specific epoch
        epoch_dir = animations_dir / f"epoch_{epoch}_{step}steps_{fps}fps"
        video_path = epoch_dir / "animation.mp4"

    if video_path.exists():
        return str(video_path)

    return None


def create_video_for_sequence(
    crop: str,
    sequence_name: str,
    sequence_path: str,
    epoch: str,
    fps: int,
    step: str,
    multi_step: str,
    overwrite: bool,
    no_interpolation: bool = False,
) -> str:
    """
    Create animation video for a sequence using create_animation.py.

    Returns:
        Path to created video, or None if creation failed
    """
    print(f"\n{'='*80}")
    print(f"Creating video for {crop}/{sequence_name}")
    print(f"{'='*80}")

    # Build the object_name path
    object_name = f"datasets/Pheno4D/processed/v1/plant_sequences/{crop}/{sequence_name}"

    # Find the experiment folder
    exp_folder = find_latest_experiment(sequence_path)
    if not exp_folder:
        print(f"  Error: No experiment folder found for {sequence_name}")
        return None

    experiment_name = Path(exp_folder).name

    # Determine num_frames based on interpolation setting
    if no_interpolation:
        num_frames = "8"  # Small number to trigger actual scan times mode in create_animation.py
        print("  Mode: Using actual scan times only (no interpolation)")
    else:
        num_frames = "100"  # Use interpolated frames for smooth video
        print("  Mode: Using interpolated frames")

    # Build command to run create_animation.py
    cmd = [
        "python",
        "create_animation.py",
        "--dataset_name",
        "pheno4d",
        "--object_name",
        object_name,
        "--experiment_name",
        experiment_name,
        "--methods",
        "ode_solve",
        "--epoch",
        epoch,
        "--step",
        step,
        "--multi_step",
        multi_step,
        "--fps",
        str(fps),
        "--num_frames",
        num_frames,
    ]

    if overwrite:
        cmd.append("--overwrite")

    print(f"  Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        # Find the created video
        video_path = find_video_for_sequence(sequence_path, epoch, fps, step)
        if video_path:
            print(f"  ✓ Video created: {video_path}")
            return video_path
        else:
            print(f"  ✗ Video creation completed but file not found")
            return None

    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error creating video for {sequence_name}:")
        print(e.stderr)
        return None


def add_text_to_video(
    input_video: str, output_video: str, text: str, position: str = "top"
) -> bool:
    """
    Add text overlay to a video using ffmpeg.

    Args:
        input_video: Path to input video
        output_video: Path to output video
        text: Text to overlay
        position: 'top' or 'bottom'

    Returns:
        True if successful
    """
    # Position the text
    if position == "top":
        y_pos = "50"
    else:
        y_pos = "h-th-50"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_video,
        "-vf",
        f"drawtext=text='{text}':fontcolor=white:fontsize=48:box=1:boxcolor=black@0.7:boxborderw=10:x=(w-text_w)/2:y={y_pos}",
        "-codec:a",
        "copy",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        output_video,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error adding text to video: {e}")
        return False


def stitch_videos(
    video_paths: List[Tuple[str, str, str]],
    output_path: str,
    grid_cols: int = 3,
) -> bool:
    """
    Stitch multiple videos into a grid layout.

    Args:
        video_paths: List of (crop, sequence_name, video_path) tuples
        output_path: Path for output video
        grid_cols: Number of columns in grid

    Returns:
        True if successful
    """
    if not video_paths:
        print("No videos to stitch")
        return False

    print(f"\n{'='*80}")
    print(f"Stitching {len(video_paths)} videos into grid ({grid_cols} columns)")
    print(f"{'='*80}")

    # Create a temporary directory for labeled videos
    temp_dir = Path("temp_labeled_videos")
    temp_dir.mkdir(exist_ok=True)

    # Add labels to each video
    labeled_videos = []
    for i, (crop, seq_name, video_path) in enumerate(video_paths):
        label = f"{crop.upper()} - {seq_name.replace('_', ' ').title()}"
        temp_video = temp_dir / f"labeled_{i:02d}.mp4"

        print(f"  Adding label to video {i+1}/{len(video_paths)}: {label}")

        if add_text_to_video(video_path, str(temp_video), label):
            labeled_videos.append(str(temp_video))
        else:
            print(f"  Warning: Failed to add label, using original video")
            labeled_videos.append(video_path)

    # Build ffmpeg filter for grid layout
    n_videos = len(labeled_videos)
    n_rows = (n_videos + grid_cols - 1) // grid_cols  # Ceiling division

    # Build input arguments
    inputs = []
    for video in labeled_videos:
        inputs.extend(["-i", video])

    # Build filter complex for grid
    # First, scale all videos to the same size
    scale_filters = []
    for i in range(n_videos):
        scale_filters.append(f"[{i}:v]scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]")

    # Build grid layout using xstack
    xstack_layout = []
    for row in range(n_rows):
        row_videos = []
        for col in range(grid_cols):
            idx = row * grid_cols + col
            if idx < n_videos:
                row_videos.append(f"v{idx}")
            else:
                # Pad with black if we don't have enough videos
                row_videos.append("black")

    # Create xstack input string
    # For a 3x4 grid (12 videos), the layout is:
    # 0_0|w0_0|w0+w1_0|w0+w1+w2_0
    # Where each position is x_y coordinate
    layout_str = ""
    for row in range(n_rows):
        for col in range(grid_cols):
            x_pos = f"w{row*grid_cols}*{col}" if col > 0 else "0"
            y_pos = f"h{0}*{row}" if row > 0 else "0"
            layout_str += f"{x_pos}_{y_pos}"
            if not (row == n_rows - 1 and col == grid_cols - 1):
                layout_str += "|"

    # Simpler approach: use hstack and vstack
    # Build horizontal stacks for each row
    filter_parts = scale_filters.copy()

    if n_rows == 1:
        # Single row - just hstack
        hstack_inputs = "+".join([f"[v{i}]" for i in range(n_videos)])
        filter_parts.append(f"{hstack_inputs}hstack=inputs={n_videos}[v]")
    else:
        # Multiple rows - create hstacks then vstack them
        for row in range(n_rows):
            start_idx = row * grid_cols
            end_idx = min(start_idx + grid_cols, n_videos)
            n_in_row = end_idx - start_idx

            if n_in_row == 1:
                # Single video in row, just pass through
                filter_parts.append(f"[v{start_idx}]copy[row{row}]")
            elif n_in_row < grid_cols and row == n_rows - 1:
                # Last row might be incomplete - pad with black
                hstack_inputs = "+".join([f"[v{i}]" for i in range(start_idx, end_idx)])
                # Add black padding
                for _ in range(grid_cols - n_in_row):
                    hstack_inputs += "+[black]"
                filter_parts.append(
                    f"color=black:s=640x480:d=10[black];{hstack_inputs}hstack=inputs={grid_cols}[row{row}]"
                )
            else:
                hstack_inputs = "+".join([f"[v{i}]" for i in range(start_idx, end_idx)])
                filter_parts.append(f"{hstack_inputs}hstack=inputs={n_in_row}[row{row}]")

        # Vstack all rows
        vstack_inputs = "+".join([f"[row{i}]" for i in range(n_rows)])
        filter_parts.append(f"{vstack_inputs}vstack=inputs={n_rows}[v]")

    filter_complex = ";".join(filter_parts)

    # Build final ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",
    ] + inputs + [
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]

    print(f"\n  Creating stitched video: {output_path}")
    print(f"  Grid layout: {n_rows} rows x {grid_cols} columns")

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n  ✓ Stitched video created: {output_path}")

        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir)

        return True

    except subprocess.CalledProcessError as e:
        print(f"\n  ✗ Error stitching videos: {e}")
        return False


def main():
    args = build_args()

    # Find all sequences
    print(f"Searching for sequences in {args.results_dir}...")
    sequences = find_sequences(args.results_dir, args.crop)

    if not sequences:
        print("No sequences found!")
        return

    print(f"Found {len(sequences)} sequences:")
    for crop, seq_name, _ in sequences:
        print(f"  - {crop}/{seq_name}")

    # Step 1: Create videos for each sequence
    video_paths = []

    if not args.no_create:
        print(f"\n{'='*80}")
        print("STEP 1: Creating individual videos")
        print(f"{'='*80}")

        for crop, seq_name, seq_path in sequences:
            # Check if video already exists
            existing_video = find_video_for_sequence(
                seq_path, args.epoch, args.fps, args.step
            )

            if existing_video and not args.overwrite:
                print(f"\n  ✓ Video already exists for {crop}/{seq_name}: {existing_video}")
                video_paths.append((crop, seq_name, existing_video))
            else:
                video_path = create_video_for_sequence(
                    crop,
                    seq_name,
                    seq_path,
                    args.epoch,
                    args.fps,
                    args.step,
                    args.multi_step,
                    args.overwrite,
                    args.no_interpolation,
                )

                if video_path:
                    video_paths.append((crop, seq_name, video_path))
                else:
                    print(f"  ✗ Failed to create video for {crop}/{seq_name}")

    else:
        # Just find existing videos
        print(f"\n{'='*80}")
        print("Finding existing videos...")
        print(f"{'='*80}")

        for crop, seq_name, seq_path in sequences:
            existing_video = find_video_for_sequence(
                seq_path, args.epoch, args.fps, args.step
            )

            if existing_video:
                print(f"  ✓ Found video for {crop}/{seq_name}: {existing_video}")
                video_paths.append((crop, seq_name, existing_video))
            else:
                print(f"  ✗ No video found for {crop}/{seq_name}")

    # Step 2: Stitch videos together
    if not args.no_stitch and video_paths:
        print(f"\n{'='*80}")
        print("STEP 2: Stitching videos together")
        print(f"{'='*80}")

        output_path = Path(args.results_dir).parent / args.output_name

        if args.group_by_crop:
            # Group videos by crop
            from collections import defaultdict
            videos_by_crop = defaultdict(list)
            for crop, seq_name, video_path in video_paths:
                videos_by_crop[crop].append((crop, seq_name, video_path))

            print(f"\n  Group-by-crop mode enabled")
            print(f"  Found {len(videos_by_crop)} crops: {list(videos_by_crop.keys())}")

            # Create a temporary directory for crop-level stitched videos
            temp_crop_dir = Path("temp_crop_videos")
            temp_crop_dir.mkdir(exist_ok=True)

            # Stitch videos for each crop
            crop_videos = []
            for crop in sorted(videos_by_crop.keys()):
                crop_video_list = videos_by_crop[crop]
                print(f"\n  Stitching {len(crop_video_list)} videos for crop: {crop}")

                crop_output = temp_crop_dir / f"{crop}_stitched.mp4"

                # Stitch all videos from this crop into one video
                # Use horizontal stacking (1 row with all videos)
                if stitch_videos(crop_video_list, str(crop_output), grid_cols=len(crop_video_list)):
                    crop_videos.append((crop, crop, str(crop_output)))
                else:
                    print(f"  ✗ Failed to stitch videos for crop: {crop}")

            # Now stitch all crop videos into a 2x2 grid
            if crop_videos:
                print(f"\n  Stitching {len(crop_videos)} crop videos into 2x2 grid")
                stitch_videos(crop_videos, str(output_path), grid_cols=2)

                # Clean up temporary crop videos
                import shutil
                shutil.rmtree(temp_crop_dir)
        else:
            # Regular stitching (all videos in one grid)
            stitch_videos(video_paths, str(output_path), args.grid_cols)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total sequences: {len(sequences)}")
    print(f"Videos created/found: {len(video_paths)}")
    print(f"Failed: {len(sequences) - len(video_paths)}")

    if not args.no_stitch and video_paths:
        print(f"\nStitched video: {output_path}")

    print("\nIndividual videos:")
    for crop, seq_name, video_path in video_paths:
        print(f"  - {crop}/{seq_name}: {video_path}")


if __name__ == "__main__":
    main()
