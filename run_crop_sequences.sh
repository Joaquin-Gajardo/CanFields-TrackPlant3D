#!/bin/bash
# Script to run all sequences for a given crop
# Usage: bash run_crop_sequences.sh <crop_name>
# Example: sbatch run_euler.sh maize

# Define sequences for each crop
# Format: "sequence_type/crop/sequence_name"
# Edit these lists based on your analyze_leaf_tips.py output
declare -A CROP_SEQUENCES

CROP_SEQUENCES[maize]="
plant_sequences/maize/maize_control_plant1
plant_sequences/maize/maize_control_plant1
plant_sequences/maize/maize_control_plant1
"

CROP_SEQUENCES[sorghum]="
plant_sequences/sorghum/sorghum_control_plant1
plant_sequences/sorghum/sorghum_highlight_plant1
plant_sequences/sorghum/sorghum_highlight_plant1
"

CROP_SEQUENCES[tobacco]="
plant_sequences/tobacco/tobacco_control_plant1
plant_sequences/tobacco/tobacco_coshade_control_plant1
plant_sequences/tobacco/tobacco_shade_plant1
"

CROP_SEQUENCES[tomato]="
plant_sequences/tomato/tomato1_heat_plant1
plant_sequences/tomato/tomato1_control2_plant1
plant_sequences/tomato/tomato2_control_plant1
"

# Get crop name and method from arguments
CROP=$1
METHOD=${2:-"ode_solve"}

if [ -z "$CROP" ]; then
    echo "Error: No crop specified"
    echo "Usage: bash run_crop_sequences.sh <crop_name> [method]"
    echo "Available crops: maize, sorghum, tobacco, tomato"
    echo "Default method: ode_solve, from options: "ode_solve", "4DSDF", "TPSR", "NVFI", "OF4D""
    exit 1
fi

# Check if crop exists
if [ -z "${CROP_SEQUENCES[$CROP]}" ]; then
    echo "Error: Unknown crop: $CROP"
    echo "Available crops: maize, sorghum, tobacco, tomato"
    exit 1
fi

# Validate all paths exist
for SEQUENCE in ${CROP_SEQUENCES[$CROP]}; do
    [ -z "$SEQUENCE" ] && continue
    SEQUENCE_PATH="datasets/Pheno4D/processed/v1/$SEQUENCE"
    if [ ! -d "$SEQUENCE_PATH" ]; then
        echo "Error: Path does not exist: $SEQUENCE_PATH"
        exit 1
    fi
done

echo "================================================================="
echo "Processing all sequences for crop: $CROP"
echo "Using method: $METHOD"
echo "================================================================="

# Loop through sequences for this crop
COUNTER=0
for SEQUENCE in ${CROP_SEQUENCES[$CROP]}; do
    # Skip empty lines
    if [ -z "$SEQUENCE" ]; then
        continue
    fi

    COUNTER=$((COUNTER + 1))
    SEQUENCE_PATH="datasets/Pheno4D/processed/v1/$SEQUENCE"
    SEQUENCE_NAME=$(basename $SEQUENCE)

    echo ""
    echo "[$COUNTER] Processing: $SEQUENCE_NAME"
    echo "    Path: $SEQUENCE_PATH"
    echo "-----------------------------------------------------------------"

    # Run the training and evaluation script
    bash run_train_eval.sh "$SEQUENCE_PATH" "$METHOD"

    echo "-----------------------------------------------------------------"
    echo "[$COUNTER] Completed: $SEQUENCE_NAME"
done

echo ""
echo "================================================================="
echo "All sequences for $CROP completed!"
echo "================================================================="
    