#!/bin/bash
# Interpolation version of run_train_eval.sh
# Usage: bash run_train_eval_interp.sh <sequence_path> [method] [interp_stride]
# Example: bash run_train_eval_interp.sh datasets/Pheno4D/.../sequence OF4D 2

# Get sequence path, method, and interp_stride from command line arguments
SEQUENCE_PATH=${1:-"datasets/Pheno4D/processed/v2/organ_sequences/tomato/tomato2_control_plant2_leaf1"}
METHOD=${2:-"OF4D"}
INTERP_STRIDE=${3:-2}
EPOCHS=15000

echo "================================================================="
echo "INTERPOLATION EXPERIMENT"
echo "================================================================="
echo "Using sequence path: $SEQUENCE_PATH"
echo "Using method: $METHOD, from options: \"ode_solve\", \"4DSDF\", \"TPSR\", \"NVFI\", \"OF4D\""
echo "Interpolation stride: $INTERP_STRIDE (train on positions 0,${INTERP_STRIDE},...)}"
echo ""

# Set adjust flag (-u) to False for non-ode_solve methods
if [ "$METHOD" != "ode_solve" ]; then
    ADJUST="False"
else
    ADJUST="True"
fi
echo "Using adjust ( -u ): $ADJUST"

# Extract crop and sequence name from path
CROP=$(echo $SEQUENCE_PATH | awk -F'/' '{print $(NF-1)}')
SEQUENCE_NAME=$(echo $SEQUENCE_PATH | awk -F'/' '{print $NF}')

# Count timesteps from dataset
NUM_TIMESTEPS=$(ls $SEQUENCE_PATH/sparse_pc 2>/dev/null | wc -l)
if [ $NUM_TIMESTEPS -eq 0 ]; then
    NUM_TIMESTEPS=$(ls $SEQUENCE_PATH/dense_pc 2>/dev/null | wc -l)
fi
echo "Detected $NUM_TIMESTEPS timesteps for $SEQUENCE_NAME (crop: $CROP)"

# Check for point split file
SPLIT_FILE="configs/point_splits/${SEQUENCE_NAME}_split.json"
if [ -f "$SPLIT_FILE" ]; then
    echo "Found point split file: $SPLIT_FILE"
    SPLIT_ARG="--point_split_file $SPLIT_FILE"
else
    echo "No point split file found, will use all points"
    SPLIT_ARG=""
fi

echo ""
echo "TRAINING CONFIGURATION:"
echo "  - Loading ALL $NUM_TIMESTEPS timesteps"
echo "  - Training on positions: 0, $INTERP_STRIDE, $((INTERP_STRIDE*2)), ... (every ${INTERP_STRIDE}th)"
echo "  - Testing on positions: 1, $((INTERP_STRIDE+1)), $((INTERP_STRIDE*2+1)), ... (interpolated)"
echo "================================================================="
echo ""

# Train CanFields with interpolation mode
python train_debugV2.py \
    -a 0.05,0.05,0.01 \
    -o $SEQUENCE_PATH \
    -d pheno4d \
    -u $ADJUST \
    -m False \
    -c 0,$NUM_TIMESTEPS,1 \
    -ev $EPOCHS \
    --method $METHOD \
    --interp_stride $INTERP_STRIDE \
    $SPLIT_ARG

# Check if training succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Training failed. Skipping evaluation."
    exit 1
fi

# Wait for filesystem to sync (helps with NFS)
echo "Training completed. Waiting 10 seconds for filesystem sync..."
sync
sleep 10

# Get the latest experiment directory
# New path: results/pheno4d/{method}/{crop}/{sequence_name}/
EXPERIMENT_DIR="results/pheno4d/$METHOD/$CROP/$SEQUENCE_NAME"
if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "ERROR: Experiment directory not found: $EXPERIMENT_DIR"
    echo "Training may have failed. Skipping evaluation."
    exit 1
fi

EXPERIMENT_NAME=$(ls -t "$EXPERIMENT_DIR" 2>/dev/null | head -n 1)
if [ -z "$EXPERIMENT_NAME" ]; then
    echo "ERROR: No experiment found in $EXPERIMENT_DIR"
    echo "Training may have failed. Skipping evaluation."
    exit 1
fi

echo ""
echo "Found experiment: $EXPERIMENT_NAME"
echo "Experiment name should contain: _interp_stride_${INTERP_STRIDE}"
echo ""

# Verify interpolation mode is in experiment name
if [[ "$EXPERIMENT_NAME" == *"interp_stride_${INTERP_STRIDE}"* ]]; then
    echo "✓ Confirmed: Experiment was run in interpolation mode"
else
    echo "WARNING: Experiment name doesn't contain expected interpolation marker!"
    echo "Expected: ..._interp_stride_${INTERP_STRIDE}"
    echo "Got: $EXPERIMENT_NAME"
fi

echo ""
echo "================================================================="
echo "EVALUATION (will automatically detect interpolation mode)"
echo "================================================================="
echo ""

# Eval: create canonical shape (only for TEST timesteps)
echo "Step 1: Creating canonical shape..."
python eval_sphere.py \
    -d pheno4d \
    -e all \
    -s 1 \
    -o $SEQUENCE_PATH \
    -m False \
    --experiment_name $EXPERIMENT_NAME \
    --method $METHOD

# Eval: deform canonical shape (only for TEST timesteps)
echo ""
echo "Step 2: Deforming canonical shape to TEST timesteps..."
python eval_sphere.py \
    -d pheno4d \
    -e all \
    -s 2 \
    -o $SEQUENCE_PATH \
    -m False \
    --experiment_name $EXPERIMENT_NAME \
    --method $METHOD

# Eval: compute metrics (Chamfer Distance and EPE) with point splits (only for TEST timesteps)
echo ""
echo "Step 3: Computing metrics on TEST timesteps with TEST points..."
python eval_metrics.py \
    -d pheno4d \
    -e all \
    -o $SEQUENCE_PATH \
    --experiment_name $EXPERIMENT_NAME \
    --methods $METHOD \
    $SPLIT_ARG

echo ""
echo "================================================================="
echo "INTERPOLATION EXPERIMENT COMPLETED"
echo "================================================================="
echo "Sequence: $SEQUENCE_NAME"
echo "Method: $METHOD"
echo "Interpolation stride: $INTERP_STRIDE"
echo "Experiment: $EXPERIMENT_NAME"
echo "================================================================="
