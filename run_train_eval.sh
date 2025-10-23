# Get sequence path and method from command line arguments
SEQUENCE_PATH=${1:-"datasets/Pheno4D/processed/v2/organ_sequences/tomato/tomato2_control_plant2_leaf1"}
METHOD=${2:-"ode_solve"}
EPOCHS=15000
echo "Using sequence path: $SEQUENCE_PATH"
echo "Using method: $METHOD, from options: \"ode_solve\", \"4DSDF\", \"TPSR\", \"NVFI\", \"OF4D\""

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

# Train CanFields
python train_debugV2.py -a 0.05,0.05,0.01 -o $SEQUENCE_PATH -d pheno4d -u $ADJUST -m False -c 0,$NUM_TIMESTEPS,1 -ev $EPOCHS --method $METHOD

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

# Eval: compute metrics (Chamfer Distance and EPE)
python eval_metrics.py -d pheno4d -e all -o $SEQUENCE_PATH --experiment_name $EXPERIMENT_NAME --methods $METHOD
