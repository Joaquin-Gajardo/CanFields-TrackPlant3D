#!/bin/bash
# Script to run all sequences for a given crop
# Usage: bash run_crop_sequences.sh <crop_name> [method] [type]
# Example: bash run_crop_sequences.sh maize ode_solve plant
# Type options: plant, leaf, both (default: plant)

# Define sequences for each crop
# Format: "sequence_type/crop/sequence_name"
# Edit these lists based on your analyze_leaf_tips.py output
declare -A CROP_SEQUENCES

CROP_SEQUENCES[maize]="
plant_sequences/maize/maize_control_plant1
plant_sequences/maize/maize_control_plant2
plant_sequences/maize/maize_control_plant3
"

CROP_SEQUENCES[sorghum]="
plant_sequences/sorghum/sorghum_control_plant1
plant_sequences/sorghum/sorghum_control_plant2
plant_sequences/sorghum/sorghum_highlight_plant2
"

CROP_SEQUENCES[tobacco]="
plant_sequences/tobacco/tobacco_control_plant1
plant_sequences/tobacco/tobacco_control_plant2
plant_sequences/tobacco/tobacco_shade_plant3
"

CROP_SEQUENCES[tomato]="
plant_sequences/tomato/tomato1_heat_plant3
plant_sequences/tomato/tomato1_shade_plant1
plant_sequences/tomato/tomato2_control_plant3
"

# Define leaf sequences for each crop (organ_sequences)
declare -A CROP_LEAF_SEQUENCES

CROP_LEAF_SEQUENCES[maize]="
organ_sequences/maize/maize_control_plant1_leaf1
organ_sequences/maize/maize_control_plant1_leaf2
organ_sequences/maize/maize_control_plant1_leaf3
organ_sequences/maize/maize_control_plant2_leaf1
organ_sequences/maize/maize_control_plant2_leaf2
organ_sequences/maize/maize_control_plant2_leaf3
organ_sequences/maize/maize_control_plant2_leaf4
organ_sequences/maize/maize_control_plant3_leaf1
organ_sequences/maize/maize_control_plant3_leaf2
organ_sequences/maize/maize_control_plant3_leaf3
"

CROP_LEAF_SEQUENCES[sorghum]="
organ_sequences/sorghum/sorghum_control_plant1_leaf1
organ_sequences/sorghum/sorghum_control_plant1_leaf2
organ_sequences/sorghum/sorghum_control_plant1_leaf3
organ_sequences/sorghum/sorghum_control_plant2_leaf1
organ_sequences/sorghum/sorghum_control_plant2_leaf2
organ_sequences/sorghum/sorghum_control_plant2_leaf3
organ_sequences/sorghum/sorghum_highlight_plant2_leaf1
organ_sequences/sorghum/sorghum_highlight_plant2_leaf2
"

CROP_LEAF_SEQUENCES[tobacco]="
organ_sequences/tobacco/tobacco_control_plant1_leaf1
organ_sequences/tobacco/tobacco_control_plant1_leaf2
organ_sequences/tobacco/tobacco_control_plant1_leaf3
organ_sequences/tobacco/tobacco_control_plant1_leaf4
organ_sequences/tobacco/tobacco_control_plant2_leaf1
organ_sequences/tobacco/tobacco_control_plant2_leaf2
organ_sequences/tobacco/tobacco_control_plant2_leaf3
organ_sequences/tobacco/tobacco_control_plant2_leaf4
organ_sequences/tobacco/tobacco_control_plant2_leaf5
organ_sequences/tobacco/tobacco_shade_plant3_leaf1
organ_sequences/tobacco/tobacco_shade_plant3_leaf2
organ_sequences/tobacco/tobacco_shade_plant3_leaf3
organ_sequences/tobacco/tobacco_shade_plant3_leaf4
organ_sequences/tobacco/tobacco_shade_plant3_leaf5
"

CROP_LEAF_SEQUENCES[tomato]="
organ_sequences/tomato/tomato1_heat_plant3_leaf1
organ_sequences/tomato/tomato1_heat_plant3_leaf2
organ_sequences/tomato/tomato1_heat_plant3_leaf3
organ_sequences/tomato/tomato1_shade_plant1_leaf1
organ_sequences/tomato/tomato1_shade_plant1_leaf2
organ_sequences/tomato/tomato1_shade_plant1_leaf3
organ_sequences/tomato/tomato1_shade_plant1_leaf4
organ_sequences/tomato/tomato1_shade_plant1_leaf5
organ_sequences/tomato/tomato2_control_plant3_leaf1
organ_sequences/tomato/tomato2_control_plant3_leaf2
organ_sequences/tomato/tomato2_control_plant3_leaf3
organ_sequences/tomato/tomato2_control_plant3_leaf4
"

# Get crop name, method, and type from arguments
CROP=$1
METHOD=${2:-"ode_solve"}
TYPE=${3:-"plant"}

if [ -z "$CROP" ]; then
    echo "Error: No crop specified"
    echo "Usage: bash run_crop_sequences.sh <crop_name> [method] [type]"
    echo "Available crops: maize, sorghum, tobacco, tomato"
    echo "Default method: ode_solve, from options: "ode_solve", "4DSDF", "TPSR", "NVFI", "OF4D""
    echo "Default type: plant, from options: plant, leaf"
    exit 1
fi

# Validate type parameter
if [[ "$TYPE" != "plant" && "$TYPE" != "leaf" ]]; then
    echo "Error: Invalid type: $TYPE"
    echo "Valid types: plant, leaf"
    exit 1
fi

# Select the appropriate sequence list based on type
if [ "$TYPE" == "leaf" ]; then
    SEQUENCES="${CROP_LEAF_SEQUENCES[$CROP]}"
    TYPE_NAME="leaf"
else
    SEQUENCES="${CROP_SEQUENCES[$CROP]}"
    TYPE_NAME="plant"
fi

# Check if sequences exist for this crop
if [ -z "$SEQUENCES" ]; then
    echo "Error: No ${TYPE_NAME} sequences defined for crop: $CROP"
    exit 1
fi

# Validate all paths exist
for SEQUENCE in $SEQUENCES; do
    [ -z "$SEQUENCE" ] && continue
    SEQUENCE_PATH="datasets/Pheno4D/processed/v2/$SEQUENCE"
    if [ ! -d "$SEQUENCE_PATH" ]; then
        echo "Error: Path does not exist: $SEQUENCE_PATH"
        exit 1
    fi
done

echo "================================================================="
echo "Processing all ${TYPE_NAME} sequences for crop: $CROP"
echo "Using method: $METHOD"
echo "================================================================="

# Loop through sequences for this crop
COUNTER=0
for SEQUENCE in $SEQUENCES; do
    # Skip empty lines
    if [ -z "$SEQUENCE" ]; then
        continue
    fi

    COUNTER=$((COUNTER + 1))
    SEQUENCE_PATH="datasets/Pheno4D/processed/v2/$SEQUENCE"
    SEQUENCE_NAME=$(basename $SEQUENCE)

    echo ""
    echo "[$COUNTER] Processing: $SEQUENCE_NAME"
    echo "    Path: $SEQUENCE_PATH"
    echo "-----------------------------------------------------------------"

    # Run the training and evaluation script
    # Continue even if this sequence fails (e.g., due to stale file handle)
    if bash run_train_eval.sh "$SEQUENCE_PATH" "$METHOD"; then
        echo "-----------------------------------------------------------------"
        echo "[$COUNTER] Completed: $SEQUENCE_NAME"
    else
        EXIT_CODE=$?
        echo "-----------------------------------------------------------------"
        echo "[$COUNTER] FAILED: $SEQUENCE_NAME (exit code: $EXIT_CODE)"
        echo "    Continuing with next sequence..."
    fi
done

echo ""
echo "================================================================="
echo "All ${TYPE_NAME} sequences for $CROP completed!"
echo "================================================================="
    