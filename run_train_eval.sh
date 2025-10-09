# Get sequence path from command line argument
SEQUENCE_PATH=${1:-"datasets/Pheno4D/processed/v1/organ_sequences/tomato/tomato2_control_plant2_leaf1"}
METHOD="ode_solve"
EPOCHS=15000
echo "Using sequence path: $SEQUENCE_PATH"

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
python train_debugV2.py -a 0.05,0.05,0.01 -o $SEQUENCE_PATH -d pheno4d -u True -m False -c 0,$NUM_TIMESTEPS,1 -ev $EPOCHS

# Get the latest experiment directory
# New path: results/pheno4d/{method}/{crop}/{sequence_name}/
EXPERIMENT_NAME=$(ls -t results/pheno4d/$METHOD/$CROP/$SEQUENCE_NAME | head -n 1)

# Eval: create canonical shape
python eval_sphere.py -d pheno4d -e all -s 1 -o $SEQUENCE_PATH -m False --experiment_name $EXPERIMENT_NAME

# Eval: deform canonical shape
python eval_sphere.py -d pheno4d -e all -s 2 -o $SEQUENCE_PATH -m False --experiment_name $EXPERIMENT_NAME
