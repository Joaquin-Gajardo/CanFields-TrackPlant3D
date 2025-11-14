#!/bin/bash
# Interpolation version of run_crop_sequences.sh
# Script to run interpolation experiments for sequences
# Usage: bash run_crop_sequences_interp.sh <crop_name> [method] [interp_stride] [type] [split] [sequence1] [sequence2] ...
# Example: bash run_crop_sequences_interp.sh maize OF4D 2 leaf test
# Example: bash run_crop_sequences_interp.sh maize OF4D 2 leaf test maize_control_plant1_leaf1
# Example: bash run_crop_sequences_interp.sh all OF4D 2 plant test  # Run all plant test sequences
#
# Arguments:
#   crop_name: maize, sorghum, tobacco, tomato, all
#   method: OF4D (default), ode_solve, 4DSDF, TPSR, NVFI
#   interp_stride: 2 (default), 3, 4, ... (stride for interpolation)
#   type: leaf (default), plant
#   split: test (default), val, all
#   sequences: (optional) run only these specific sequences

# Import the sequence definitions from run_crop_sequences.sh
# (We don't actually source it since we redefine all sequences below)
# source run_crop_sequences.sh 2>/dev/null || true

# Get arguments
CROP=$1
METHOD=${2:-"OF4D"}
INTERP_STRIDE=${3:-2}
TYPE=${4:-"leaf"}
SPLIT=${5:-"test"}

# Collect all remaining arguments as specific sequences (if any)
shift 5 2>/dev/null || shift $#
SPECIFIC_SEQUENCES="$@"

if [ -z "$CROP" ]; then
    echo "Error: No crop specified"
    echo "Usage: bash run_crop_sequences_interp.sh <crop_name> [method] [interp_stride] [type] [split] [sequence1] [sequence2] ..."
    echo "Available crops: maize, sorghum, tobacco, tomato, all"
    echo "Default method: OF4D, from options: ode_solve, 4DSDF, TPSR, NVFI, OF4D"
    echo "Default interp_stride: 2"
    echo "Default type: leaf, from options: plant, leaf"
    echo "Default split: test, from options: test, val, all"
    echo "Optional sequences: run only these specific sequences"
    exit 1
fi

echo "================================================================="
echo "INTERPOLATION EXPERIMENTS"
echo "================================================================="
echo "Crop: $CROP"
echo "Method: $METHOD"
echo "Interpolation stride: $INTERP_STRIDE"
echo "Type: $TYPE"
echo "Split: $SPLIT"
if [ -n "$SPECIFIC_SEQUENCES" ]; then
    echo "Specific sequences: $SPECIFIC_SEQUENCES"
fi
echo "================================================================="
echo ""

# Use the existing run_crop_sequences.sh logic but call run_train_eval_interp.sh instead
# We'll pass the sequences through to the wrapper below

# Define sequences for each crop (copy from run_crop_sequences.sh or source it)
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

# Define leaf sequences
declare -A CROP_LEAF_SEQUENCES
CROP_LEAF_SEQUENCES[maize]="
organ_sequences/maize/maize_control_plant1_leaf1
organ_sequences/maize/maize_control_plant1_leaf2
organ_sequences/maize/maize_control_plant1_leaf3
organ_sequences/maize/maize_control_plant1_leaf4
organ_sequences/maize/maize_control_plant2_leaf1
organ_sequences/maize/maize_control_plant2_leaf2
organ_sequences/maize/maize_control_plant2_leaf3
organ_sequences/maize/maize_control_plant2_leaf4
organ_sequences/maize/maize_control_plant2_leaf5
organ_sequences/maize/maize_control_plant3_leaf1
organ_sequences/maize/maize_control_plant3_leaf2
organ_sequences/maize/maize_control_plant3_leaf3
organ_sequences/maize/maize_control_plant3_leaf4
"

CROP_LEAF_SEQUENCES[sorghum]="
organ_sequences/sorghum/sorghum_control_plant1_leaf1
organ_sequences/sorghum/sorghum_control_plant1_leaf2
organ_sequences/sorghum/sorghum_control_plant1_leaf3
organ_sequences/sorghum/sorghum_control_plant2_leaf1
organ_sequences/sorghum/sorghum_control_plant2_leaf2
organ_sequences/sorghum/sorghum_control_plant2_leaf3
organ_sequences/sorghum/sorghum_control_plant2_leaf4
organ_sequences/sorghum/sorghum_highlight_plant2_leaf1
organ_sequences/sorghum/sorghum_highlight_plant2_leaf2
organ_sequences/sorghum/sorghum_highlight_plant2_leaf3
organ_sequences/sorghum/sorghum_highlight_plant2_leaf4
organ_sequences/sorghum/sorghum_highlight_plant2_leaf5
"

CROP_LEAF_SEQUENCES[tobacco]="
organ_sequences/tobacco/tobacco_control_plant1_leaf1
organ_sequences/tobacco/tobacco_control_plant1_leaf2
organ_sequences/tobacco/tobacco_control_plant1_leaf3
organ_sequences/tobacco/tobacco_control_plant1_leaf4
organ_sequences/tobacco/tobacco_control_plant1_leaf5
organ_sequences/tobacco/tobacco_control_plant1_leaf6
organ_sequences/tobacco/tobacco_control_plant2_leaf1
organ_sequences/tobacco/tobacco_control_plant2_leaf2
organ_sequences/tobacco/tobacco_control_plant2_leaf3
organ_sequences/tobacco/tobacco_control_plant2_leaf4
organ_sequences/tobacco/tobacco_control_plant2_leaf5
organ_sequences/tobacco/tobacco_control_plant2_leaf6
organ_sequences/tobacco/tobacco_shade_plant3_leaf1
organ_sequences/tobacco/tobacco_shade_plant3_leaf2
organ_sequences/tobacco/tobacco_shade_plant3_leaf3
organ_sequences/tobacco/tobacco_shade_plant3_leaf4
organ_sequences/tobacco/tobacco_shade_plant3_leaf5
organ_sequences/tobacco/tobacco_shade_plant3_leaf6
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
organ_sequences/tomato/tomato1_shade_plant1_leaf6
organ_sequences/tomato/tomato1_shade_plant1_leaf7
organ_sequences/tomato/tomato1_shade_plant1_leaf8
organ_sequences/tomato/tomato2_control_plant3_leaf1
organ_sequences/tomato/tomato2_control_plant3_leaf2
organ_sequences/tomato/tomato2_control_plant3_leaf3
organ_sequences/tomato/tomato2_control_plant3_leaf4
organ_sequences/tomato/tomato2_control_plant3_leaf5
organ_sequences/tomato/tomato2_control_plant3_leaf6
organ_sequences/tomato/tomato2_control_plant3_leaf7
"

# If CROP is "all", expand to all available crops
if [ "$CROP" == "all" ]; then
    ALL_CROPS="maize sorghum tobacco tomato"
    echo "Running interpolation experiments for ALL crops: $ALL_CROPS"
    echo ""

    for SINGLE_CROP in $ALL_CROPS; do
        echo ""
        echo ">>> Starting crop: $SINGLE_CROP"
        bash "$0" "$SINGLE_CROP" "$METHOD" "$INTERP_STRIDE" "$TYPE" "$SPLIT" $SPECIFIC_SEQUENCES
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo ">>> FAILED: $SINGLE_CROP (exit code: $EXIT_CODE)"
        else
            echo ">>> COMPLETED: $SINGLE_CROP"
        fi
    done

    echo ""
    echo "================================================================="
    echo "All crops completed!"
    echo "================================================================="
    exit 0
fi

# Select sequence list based on type
if [ "$TYPE" == "leaf" ]; then
    SEQUENCES="${CROP_LEAF_SEQUENCES[$CROP]}"
else
    SEQUENCES="${CROP_SEQUENCES[$CROP]}"
fi

# Filter by specific sequences if provided
if [ -n "$SPECIFIC_SEQUENCES" ]; then
    FILTERED_SEQUENCES=""
    for REQUESTED_SEQ in $SPECIFIC_SEQUENCES; do
        for SEQUENCE in $SEQUENCES; do
            [ -z "$SEQUENCE" ] && continue
            SEQUENCE_NAME=$(basename "$SEQUENCE")
            if [ "$SEQUENCE_NAME" == "$REQUESTED_SEQ" ]; then
                FILTERED_SEQUENCES="$FILTERED_SEQUENCES
$SEQUENCE"
                break
            fi
        done
    done
    SEQUENCES="$FILTERED_SEQUENCES"
fi

# Filter by split using the function from run_crop_sequences.sh
# (We'll just use a simple version here for interpolation experiments)
VALID_SEQUENCES=""
for SEQUENCE in $SEQUENCES; do
    [ -z "$SEQUENCE" ] && continue
    SEQUENCE_PATH="datasets/Pheno4D/processed/v2/$SEQUENCE"
    if [ -d "$SEQUENCE_PATH" ]; then
        VALID_SEQUENCES="$VALID_SEQUENCES
$SEQUENCE"
    fi
done
SEQUENCES="$VALID_SEQUENCES"

# Count sequences
VALID_COUNT=$(echo "$SEQUENCES" | sed '/^$/d' | wc -l)
if [ $VALID_COUNT -eq 0 ]; then
    echo "Error: No valid sequences found"
    exit 1
fi

echo "Processing $VALID_COUNT sequences"
echo ""

# Loop through sequences
COUNTER=0
for SEQUENCE in $SEQUENCES; do
    [ -z "$SEQUENCE" ] && continue

    COUNTER=$((COUNTER + 1))
    SEQUENCE_PATH="datasets/Pheno4D/processed/v2/$SEQUENCE"
    SEQUENCE_NAME=$(basename $SEQUENCE)

    echo ""
    echo "================================================================="
    echo "[$COUNTER/$VALID_COUNT] $SEQUENCE_NAME"
    echo "================================================================="

    # Run interpolation training and evaluation
    if bash run_train_eval_interp.sh "$SEQUENCE_PATH" "$METHOD" "$INTERP_STRIDE"; then
        echo "✓ Completed: $SEQUENCE_NAME"
    else
        EXIT_CODE=$?
        echo "✗ FAILED: $SEQUENCE_NAME (exit code: $EXIT_CODE)"
        echo "Continuing with next sequence..."
    fi
done

echo ""
echo "================================================================="
echo "Interpolation experiments completed for $CROP!"
echo "================================================================="
