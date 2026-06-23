# Get sequence path, method and optional run index from command line arguments
SEQUENCE_PATH=${1:-"datasets/Pheno4D/processed/v2/organ_sequences/tomato/tomato2_control_plant2_leaf1"}
METHOD=${2:-"ode_solve"}
RUN_IDX=${3:-""}  # Optional 1-indexed run number for repeated runs of the same sequence
VELO_EPOCH=7000   # iteration at which velocity-network regularizers turn on (paper: 7000/15000)
echo "Using sequence path: $SEQUENCE_PATH"
echo "Using method: $METHOD, from options: \"ode_solve\", \"4DSDF\", \"TPSR\", \"NVFI\", \"OF4D\""

# Build optional --run_idx argument so repeated runs land in separate experiment dirs
if [ -n "$RUN_IDX" ]; then
    RUN_ARG="--run_idx $RUN_IDX"
    echo "Using run index: $RUN_IDX"
else
    RUN_ARG=""
fi

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

# Train CanFields
python train_debugV2.py -a 0.05,0.05,0.01 -o $SEQUENCE_PATH -d pheno4d -u $ADJUST -m False -c 0,$NUM_TIMESTEPS,1 -ev $VELO_EPOCH --method $METHOD $SPLIT_ARG $RUN_ARG

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

echo "Found experiment: $EXPERIMENT_NAME"

# Eval: create canonical shape
python eval_sphere.py -d pheno4d -e all -s 1 -o $SEQUENCE_PATH -m False --experiment_name $EXPERIMENT_NAME --method $METHOD

# Eval: deform canonical shape
python eval_sphere.py -d pheno4d -e all -s 2 -o $SEQUENCE_PATH -m False --experiment_name $EXPERIMENT_NAME --method $METHOD

# Eval: compute metrics (Chamfer Distance and EPE) with point splits
python eval_metrics.py -d pheno4d -e all -o $SEQUENCE_PATH --experiment_name $EXPERIMENT_NAME --methods $METHOD $SPLIT_ARG
