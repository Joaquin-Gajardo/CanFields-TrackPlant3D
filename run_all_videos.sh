#!/bin/bash
# Quick script to create videos for all plants and stitch them together
# Usage examples:
#   bash run_all_videos.sh                    # Use last epoch, 2 fps
#   bash run_all_videos.sh 15000 5            # Use epoch 15000, 5 fps
#   bash run_all_videos.sh last 2 maize       # Only maize sequences

EPOCH=${1:-"last"}
FPS=${2:-2}
CROP=${3:-""}

echo "================================================================="
echo "Creating videos for all plant sequences"
echo "Epoch: $EPOCH"
echo "FPS: $FPS"
if [ -n "$CROP" ]; then
    echo "Crop filter: $CROP"
fi
echo "================================================================="

# Build command
CMD="python create_all_videos_and_stitch.py --epoch $EPOCH --fps $FPS"

if [ -n "$CROP" ]; then
    CMD="$CMD --crop $CROP"
fi

# Run the command
echo "Running: $CMD"
echo ""

$CMD
