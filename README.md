# CanFields: Consolidating Diffeomorphic Flows for Non-Rigid 4D Interpolation from Arbitrary-Length Sequences (ICCV2025)

### [[Project Page](https://wangmiaowei.github.io/CanFields.github.io/)] [[arXiv](https://arxiv.org/abs/2406.18582)] 

Miaowei Wang, Changjian Li, Amir Vaxman <br>

The University of Edinburgh <br>

![CanFields](assets/teaser.png)
---




### Abstract

*We introduce Canonical Consolidation Fields (CanFields). This novel method interpolates arbitrary-length sequences of independently sampled 3D point clouds into a unified, continuous, and coherent deforming shape. Unlike prior methods that oversmooth geometry or produce topological and geometric artifacts, CanFields optimizes fine-detailed geometry and deformation jointly in an unsupervised fitting with two novel bespoke modules. First, we introduce a dynamic consolidator module that adjusts the input and assigns confidence scores, balancing the optimization of the canonical shape and its motion. Second, we represent the motion as a diffeomorphic flow parameterized by a smooth velocity field. We have validated our robustness and accuracy on more than 50 diverse sequences, demonstrating its superior performance even with missing regions, noisy raw scans, and sparse data.*

---

## Environment Setup

### Prerequisites
- NVIDIA GPU with CUDA support
- Miniconda or Anaconda installed

### Installation Steps

```bash
# Navigate to the project directory
git clone https://github.com/wangmiaowei/CanFields.git
cd CanFields

# Create conda environment with Python 3.9
conda create -n canfields python=3.9 -y

# Activate the environment
conda activate canfields

# Get gcc and g++ compilers for Pytorch3D
conda install -c conda-forge gcc gxx

# Install pytorch, pytorch3d, torchmeta, and other dependencies
pip install -r requirements_updated.txt

# Alternatively, install dependencies step-by-step as below
# # Install PyTorch 1.13.0 with CUDA 11.6 support (see https://github.com/wangmiaowei/CanFields/issues/2#issuecomment-3027513665)
# pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# # Install compatible PyTorch3D from source (version 0.7.2). See https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
# pip install pytorch3d@https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1130/pytorch3d-0.7.2-cp39-cp39-linux_x86_64.whl

# # Install torchmeta 1.8.0 from fork (original requires torch<1.10.0 and torchvision<0.11.0)
# pip install git+https://github.com/Joaquin-Gajardo/pytorch-meta.git@1.8.0-torchvision-fix

# # Install remaining dependencies with compatible versions
# pip install GPUtil==1.4.0 numpy==1.22.4 plyfile==1.1 point_cloud_utils==0.30.4 pynvml==11.5.0 pytest==8.1.1 scikit_learn==1.3.2 scipy==1.7.3 scikit-image torchdiffeq==0.2.3 wandb==0.16.3 trimesh open3d
```

## Quick Evaluation

Our algorithm has been tested on multiple datasets. Below is the quick evaluation process for the raw scanned sequence `one_leg_loose` as presented in Fig. 10.

```bash
# Navigate to raw scanned sequence
# '4k_pc' contains sampled points as input, 'gt_mesh' consists of raw scans.
cd datasets/Deforming4D/notexture/50002_one_leg_loose_1 

# Check our reconstructed results
cd ../../../..  # Adjusting the directory path to reach root
cd results/manifold

# Access provided pretrained weights
cd ../..  # Adjusting to the weights directory
cd exp_res/formulation/ode_solve

# Step 1: Generate the consolidated canonical shape
cd ../../..
python eval_sphere.py -d animals -e 15000 -s 1 -o 50002_one_leg_loose_1_full_l2_0.05_elas_0.05_prob_0.01_start_id_0_gap_15_stride_1 -m False

# Step 2: Generate the reconstructed results
python eval_sphere.py -d animals -e 15000 -s 2 -o 50002_one_leg_loose_1_full_l2_0.05_elas_0.05_prob_0.01_start_id_0_gap_15_stride_1 -m False
```

---

## Fitting from Raw Scans

To train CanFields using sampled points from raw scans, execute:

```bash
# Train CanFields
python train_debugV2.py -a 0.05,0.05,0.01 -o 50002_one_leg_loose_1 -d animals -u True -m False -c 0,15,1 -ev 7000
```
### Pheno4D dataset

#### Data preparation
This steps will automatically download the dataset and prepare the sequences for training.
```bash
# 1. Install plant-shape-analysis package
pip install git+https://github.com/Joaquin-Gajardo/plant-shape-analysis.git

# 2. Prepare dataset (default: version 2, saves both plants and organs sequences, sparse only)
python datasets/prepare_pheno4d.py --dataset-path datasets/Pheno4D/raw --output-dir datasets/Pheno4D/processed # Defaults: --version v2 --mode both --export-sparse
````

#### Leaf sequence
```bash
# Train CanFields
python train_debugV2.py -a 0.05,0.05,0.01 -o datasets/Pheno4D/processed/v2/organ_sequences/tomato/tomato2_control_plant2_leaf1 -d pheno4d -u True -m False -c 0,9,1 -ev 15000

# Eval: create canonical shape
python eval_sphere.py -d pheno4d -e 15000 -s 1 -o datasets/Pheno4D/processed/v2/organ_sequences/tomato/tomato2_control_plant2_leaf1 -m False --experiment_name tomato2_control_plant2_leaf1_P4D_full_l2_0.05_elas_0.05_prob_0.01_start_id_0_gap_9_stride_1

# Eval: deform canonical shape
python eval_sphere.py -d pheno4d -e 15000 -s 2 -o datasets/Pheno4D/processed/v2/organ_sequences/tomato/tomato2_control_plant2_leaf1 -m False --experiment_name tomato2_control_plant2_leaf1_P4D_full_l2_0.05_elas_0.05_prob_0.01_start_id_0_gap_9_stride_1
```
#### Plant sequence
Note that this could require up to 80GB of GPU memory.
```bash
python train_debugV2.py -a 0.05,0.05,0.01 -o datasets/Pheno4D/processed/v2/plant_sequences/tomato/tomato2_control_plant2 -d pheno4d -u True -m False -c 0,9,1 -ev 15000
python eval_sphere.py -d pheno4d -e 15000 -s 1 -o datasets/Pheno4D/processed/v2/plant_sequences/tomato/tomato2_control_plant2 -m False --experiment_name {timestamp}_tomato2_control_plant2_P4D_full_l2_0.05_elas_0.05_prob_0.01_start_id_0_gap_9_stride_1
python eval_sphere.py -d pheno4d -e 15000 -s 2 -o datasets/Pheno4D/processed/v2/plant_sequences/tomato/tomato2_control_plant2 -m False --experiment_name {timestamp}_tomato2_control_plant2_P4D_full_l2_0.05_elas_0.05_prob_0.01_start_id_0_gap_9_stride_1
```

#### Animation
To animate the reconstructed sequence, run:

```bash
python create_animation.py -d pheno4d -e 15000 -m 1 -o datasets/Pheno4D/processed/v2/organ_sequences/tomato/tomato2_control_plant2_leaf1 -en {timestamp}_tomato2_control_plant2_leaf1_P4D_full_l2_0.05_elas_0.05_prob_0.01_start_id_0_gap_9_stride_1 --num_frames 9 --fps 2
```

#### Quantitative Evaluation

Compute reconstruction quality metrics (Chamfer Distance and Leaf Tip EPE):

```bash
# Compute metrics
python eval_metrics.py -d pheno4d -o <path> -en <experiment_name> -e last --num_samples 10000

# Aggregate results across experiments
python metrics/aggregate_results.py --base_dir results/pheno4d --export_csv --export_latex
```

**Metrics saved to**: `results/.../metrics/metrics_epoch_{epoch}.json`

---

## Training on ETH HPC (euler)
```bash
module purge
conda activate canfields
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy
sbatch run_euler.sh datasets/Pheno4D/processed/v2/organ_sequences/tomato/tomato2_control_plant2
```

## Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
    @InProceedings{CanFields2025,
    title={CanFields: Consolidating Diffeomorphic Flows for Non-Rigid 4D Interpolation from Arbitrary-Length Sequences},
    author={Wang, Miaowei and Li, Changjian and Vaxman, Amir},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2025}}
   ```
