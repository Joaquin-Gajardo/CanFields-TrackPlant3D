# Get object name from command line argument
OBJECT_NAME=${1:-"tomato2_control_plant2"}
DATASET_PATH="datasets/Pheno4D/plants/$OBJECT_NAME"
METHOD="ode_solve"
EPOCHS=15000

# Count timesteps from dataset
NUM_TIMESTEPS=$(ls $DATASET_PATH/sparse_pc | wc -l)
echo "Detected $NUM_TIMESTEPS timesteps in $DATASET_PATH/sparse_pc"

# Train CanFields
#python train_debugV2.py -a 0.05,0.05,0.01 -o $DATASET_PATH -d pheno4d -u True -m False -c 0,$NUM_TIMESTEPS,1 -ev $EPOCHS

# Get the latest experiment directory
EXPERIMENT_NAME=$(ls -t results/pheno4d/$OBJECT_NAME/$METHOD | head -n 1)

# Eval: create canonical shape
python eval_sphere.py -d pheno4d -e all -s 1 -o $DATASET_PATH -m False --experiment_name $EXPERIMENT_NAME

# Eval: deform canonical shape
python eval_sphere.py -d pheno4d -e all -s 2 -o $DATASET_PATH -m False --experiment_name $EXPERIMENT_NAME
