#!/bin/bash
#SBATCH -J CanFields_Interp
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:80g
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=sbatch_log/%j_interp.out
#SBATCH --error=sbatch_log/%j_interp.out

module purge

/cluster/project/cropsci/jgajardo/miniconda3/bin/conda init
conda activate base
conda activate canfields
conda list

module load stack/2024-05
module load gcc/13.2.0 cuda/12.2.1 eth_proxy
echo $(module list)

nvidia-smi

echo "================================================================="
echo "INTERPOLATION EXPERIMENT MODE"
echo "================================================================="

# Run crop sequences with interpolation
# Usage: sbatch run_euler_interp.sh <crop_name> [method] [interp_stride] [type] [split] [sequence1] [sequence2] ...
# Example: sbatch run_euler_interp.sh maize OF4D 2 leaf test
# Example: sbatch run_euler_interp.sh maize OF4D 2 leaf test maize_control_plant1_leaf1
bash run_crop_sequences_interp.sh "$@"
