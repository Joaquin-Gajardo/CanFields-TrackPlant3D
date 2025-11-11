#!/bin/bash
# Script to run all sequences for a given crop
# Usage: bash run_crop_sequences.sh <crop_name> [method] [type] [split]
# Example: bash run_crop_sequences.sh maize ode_solve leaf test
#
# Arguments:
#   crop_name: maize, sorghum, tobacco, tomato
#   method: ode_solve (default), 4DSDF, TPSR, NVFI, OF4D
#   type: plant, leaf (default: plant)
#   split: test (default), val, all
#
# Sequences match DOMA val+test splits for fair comparison:
# - Plant sequences: 12 (4 val + 8 test)
# - Organ sequences: 73 (24 val + 49 test from above 12 plants)
#   - 61 leaf sequences + 12 stem sequences
# - Point splits available in: configs/point_splits/

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
organ_sequences/maize/maize_control_plant1_leaf4
organ_sequences/maize/maize_control_plant1_stem
organ_sequences/maize/maize_control_plant2_leaf1
organ_sequences/maize/maize_control_plant2_leaf2
organ_sequences/maize/maize_control_plant2_leaf3
organ_sequences/maize/maize_control_plant2_leaf4
organ_sequences/maize/maize_control_plant2_leaf5
organ_sequences/maize/maize_control_plant2_stem
organ_sequences/maize/maize_control_plant3_leaf1
organ_sequences/maize/maize_control_plant3_leaf2
organ_sequences/maize/maize_control_plant3_leaf3
organ_sequences/maize/maize_control_plant3_leaf4
organ_sequences/maize/maize_control_plant3_stem
"

CROP_LEAF_SEQUENCES[sorghum]="
organ_sequences/sorghum/sorghum_control_plant1_leaf1
organ_sequences/sorghum/sorghum_control_plant1_leaf2
organ_sequences/sorghum/sorghum_control_plant1_leaf3
organ_sequences/sorghum/sorghum_control_plant1_stem
organ_sequences/sorghum/sorghum_control_plant2_leaf1
organ_sequences/sorghum/sorghum_control_plant2_leaf2
organ_sequences/sorghum/sorghum_control_plant2_leaf3
organ_sequences/sorghum/sorghum_control_plant2_leaf4
organ_sequences/sorghum/sorghum_control_plant2_stem
organ_sequences/sorghum/sorghum_highlight_plant2_leaf1
organ_sequences/sorghum/sorghum_highlight_plant2_leaf2
organ_sequences/sorghum/sorghum_highlight_plant2_leaf3
organ_sequences/sorghum/sorghum_highlight_plant2_leaf4
organ_sequences/sorghum/sorghum_highlight_plant2_leaf5
organ_sequences/sorghum/sorghum_highlight_plant2_stem
"

CROP_LEAF_SEQUENCES[tobacco]="
organ_sequences/tobacco/tobacco_control_plant1_leaf1
organ_sequences/tobacco/tobacco_control_plant1_leaf2
organ_sequences/tobacco/tobacco_control_plant1_leaf3
organ_sequences/tobacco/tobacco_control_plant1_leaf4
organ_sequences/tobacco/tobacco_control_plant1_leaf5
organ_sequences/tobacco/tobacco_control_plant1_leaf6
organ_sequences/tobacco/tobacco_control_plant1_stem
organ_sequences/tobacco/tobacco_control_plant2_leaf1
organ_sequences/tobacco/tobacco_control_plant2_leaf2
organ_sequences/tobacco/tobacco_control_plant2_leaf3
organ_sequences/tobacco/tobacco_control_plant2_leaf4
organ_sequences/tobacco/tobacco_control_plant2_leaf5
organ_sequences/tobacco/tobacco_control_plant2_leaf6
organ_sequences/tobacco/tobacco_control_plant2_stem
organ_sequences/tobacco/tobacco_shade_plant3_leaf1
organ_sequences/tobacco/tobacco_shade_plant3_leaf2
organ_sequences/tobacco/tobacco_shade_plant3_leaf3
organ_sequences/tobacco/tobacco_shade_plant3_leaf4
organ_sequences/tobacco/tobacco_shade_plant3_leaf5
organ_sequences/tobacco/tobacco_shade_plant3_leaf6
organ_sequences/tobacco/tobacco_shade_plant3_stem
"

# Note: Above sequences match DOMA val+test splits (73 total organ sequences)
# - Val: 24 sequences (20 leaves + 4 stems) from 4 plants
# - Test: 49 sequences (41 leaves + 8 stems) from 8 plants
# Total: 61 leaves + 12 stems = 73 organ sequences
# These are the same sequences used in DOMA experiments for fair comparison

CROP_LEAF_SEQUENCES[tomato]="
organ_sequences/tomato/tomato1_heat_plant3_leaf1
organ_sequences/tomato/tomato1_heat_plant3_leaf2
organ_sequences/tomato/tomato1_heat_plant3_leaf3
organ_sequences/tomato/tomato1_heat_plant3_stem
organ_sequences/tomato/tomato1_shade_plant1_leaf1
organ_sequences/tomato/tomato1_shade_plant1_leaf2
organ_sequences/tomato/tomato1_shade_plant1_leaf3
organ_sequences/tomato/tomato1_shade_plant1_leaf4
organ_sequences/tomato/tomato1_shade_plant1_leaf5
organ_sequences/tomato/tomato1_shade_plant1_leaf6
organ_sequences/tomato/tomato1_shade_plant1_leaf7
organ_sequences/tomato/tomato1_shade_plant1_leaf8
organ_sequences/tomato/tomato1_shade_plant1_stem
organ_sequences/tomato/tomato2_control_plant3_leaf1
organ_sequences/tomato/tomato2_control_plant3_leaf2
organ_sequences/tomato/tomato2_control_plant3_leaf3
organ_sequences/tomato/tomato2_control_plant3_leaf4
organ_sequences/tomato/tomato2_control_plant3_leaf5
organ_sequences/tomato/tomato2_control_plant3_leaf6
organ_sequences/tomato/tomato2_control_plant3_leaf7
organ_sequences/tomato/tomato2_control_plant3_stem
"

# Get crop name, method, type, and split from arguments
CROP=$1
METHOD=${2:-"ode_solve"}
TYPE=${3:-"plant"}
SPLIT=${4:-"test"}  # Default to test sequences

if [ -z "$CROP" ]; then
    echo "Error: No crop specified"
    echo "Usage: bash run_crop_sequences.sh <crop_name> [method] [type] [split]"
    echo "Available crops: maize, sorghum, tobacco, tomato"
    echo "Default method: ode_solve, from options: ode_solve, 4DSDF, TPSR, NVFI, OF4D"
    echo "Default type: plant, from options: plant, leaf"
    echo "Default split: test, from options: test, val, all"
    exit 1
fi

# Validate type parameter
if [[ "$TYPE" != "plant" && "$TYPE" != "leaf" ]]; then
    echo "Error: Invalid type: $TYPE"
    echo "Valid types: plant, leaf"
    exit 1
fi

# Validate split parameter
if [[ "$SPLIT" != "test" && "$SPLIT" != "val" && "$SPLIT" != "all" ]]; then
    echo "Error: Invalid split: $SPLIT"
    echo "Valid splits: test, val, all"
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

# Function to check if a sequence is in a specific split
is_in_split() {
    local sequence_name=$1
    local target_split=$2

    # Extract just the sequence name (e.g., "maize_control_plant1" or "maize_control_plant1_leaf1")
    local seq_basename=$(basename "$sequence_name")

    # Use Python to check if sequence is in the specified split
    python3 -c "
import json
import sys
with open('configs/data_splits.json') as f:
    splits = json.load(f)

seq_name = '$seq_basename'
target_split = '$target_split'

if target_split == 'all':
    sys.exit(0)  # Include all sequences

# Check if sequence is in the target split
sequences = splits['splits'][target_split]['sequences']
plants = splits['splits'][target_split]['plants']

# Check both exact match and if it belongs to a plant in the split
if seq_name in sequences or seq_name in plants:
    sys.exit(0)  # In split

# For plant sequences, check if plant name matches
for plant in plants:
    if seq_name == plant:
        sys.exit(0)

sys.exit(1)  # Not in split
" 2>/dev/null
    return $?
}

# Filter sequences based on split parameter
FILTERED_SEQUENCES=""
for SEQUENCE in $SEQUENCES; do
    [ -z "$SEQUENCE" ] && continue

    if is_in_split "$SEQUENCE" "$SPLIT"; then
        FILTERED_SEQUENCES="$FILTERED_SEQUENCES
$SEQUENCE"
    fi
done

# Update SEQUENCES to filtered list
SEQUENCES="$FILTERED_SEQUENCES"

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
echo "Processing ${SPLIT} ${TYPE_NAME} sequences for crop: $CROP"
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
    