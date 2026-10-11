# CanFields: Consolidating Diffeomorphic Flows for Non-Rigid 4D Interpolation from Arbitrary-Length Sequences (ICCV2025)

### [[Project Page](https://wangmiaowei.github.io/CanFields.github.io/)] [[arXiv](https://arxiv.org/abs/2406.18582)] 

Miaowei Wang, Changjian Li, Amir Vaxman <br>

The University of Edinburgh <br>

![CanFields](assets/teaser.png)
---

> **About this fork.** This is a fork of [wangmiaowei/CanFields](https://github.com/wangmiaowei/CanFields)
> used to run the NDF, NVFi, DSR and CanFields baselines of
> [GrowFields](https://github.com/Joaquin-Gajardo/growfields) (ECCV 2026) on the TrackPlant3D
> plant growth dataset. The methods are the original authors'; the additions are a TrackPlant3D
> loader, the evaluation splits and metrics, and the scripts to run them. See
> [GrowFields baselines on TrackPlant3D](#growfields-baselines-on-trackplant3d) below. The
> sections before it are the original README.


### Abstract

*We introduce Canonical Consolidation Fields (CanFields). This novel method interpolates arbitrary-length sequences of independently sampled 3D point clouds into a unified, continuous, and coherent deforming shape. Unlike prior methods that oversmooth geometry or produce topological and geometric artifacts, CanFields optimizes fine-detailed geometry and deformation jointly in an unsupervised fitting with two novel bespoke modules. First, we introduce a dynamic consolidator module that adjusts the input and assigns confidence scores, balancing the optimization of the canonical shape and its motion. Second, we represent the motion as a diffeomorphic flow parameterized by a smooth velocity field. We have validated our robustness and accuracy on more than 50 diverse sequences, demonstrating its superior performance even with missing regions, noisy raw scans, and sparse data.*

---

## Environment Setup

### Prerequisites
- NVIDIA GPU with CUDA support
- Miniconda or Anaconda installed

### Installation Steps

```bash
git clone https://github.com/Joaquin-Gajardo/CanFields.git
cd CanFields

# Create conda environment with Python 3.9
conda create -n canfields python=3.9 -y
conda activate canfields

# Get gcc and g++ compilers for Pytorch3D
conda install -c conda-forge gcc gxx

# Install pytorch, pytorch3d, torchmeta, plant-shape-analysis and other dependencies
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

---

## GrowFields baselines on TrackPlant3D

These are the commands behind the NDF, NVFi, DSR and CanFields rows of Tables 1 (full plant)
and 2 (per organ) in the GrowFields paper. The baselines were not run in the interpolation,
extrapolation or robustness experiments.

### Methods

| Paper name | `--method` | Consolidator (`-u`) |
| --- | --- | --- |
| NDF | `TPSR` | False |
| NVFi | `NVFI` | False |
| DSR | `4DSDF` | False |
| CanFields | `ode_solve` | True |

`run_train_eval.sh` sets `-u` from the method. `OF4D` is also implemented but was not used in
the paper. DSR has no EPE in the paper: it models each frame independently, so it gives no
point correspondences to track.

### What differs from upstream

- **A regularizer fix for NDF and NVFi** (`networks/other_volocities.py`). The velocity
  regularizer expects its Jacobian as `[dt, dx, dy, dz]`. NVFi's was passed with the time
  column in the wrong place, and NDF's dynamics are autonomous, so a zero time column is now
  prepended. All published runs include this fix.
- **`-ev 7000`.** `-ev` is the iteration at which the velocity regularizers switch on, not
  the number of epochs (training runs for 15,000). `run_train_eval.sh` uses 7000.
- **Seeds.** The published runs all used seed 3407, so the five runs per sequence differ only
  through GPU nondeterminism, and the baseline rows are in effect single runs. The code now seeds each run with `3407 + run_idx`. `--n_runs 1` (the default)
  uses 3407, the published seed.
- **TrackPlant3D support**: `datasets/data_pheno4d.py`, `datasets/prepare_pheno4d.py`, the
  train/test point splits in `configs/point_splits/`, and the metrics in `eval_metrics.py`
  (Chamfer distance on held-out surface points, and leaf-tip EPE matched by organ label).

### Data

`prepare_pheno4d.py` downloads TrackPlant3D v2 through
[plant-shape-analysis](https://github.com/Joaquin-Gajardo/plant-shape-analysis) (installed by
`requirements_updated.txt`) and writes plant and organ sequences under
`datasets/Pheno4D/processed/v2/`. The processed dataset is also deposited in the
[ETH Research Collection](https://hdl.handle.net/20.500.11850/806887).

```bash
# Plant and leaf sequences
python datasets/prepare_pheno4d.py \
      --dataset-path datasets/Pheno4D/raw \
      --version v2 \
      --output-dir datasets/Pheno4D/processed \
      --mode both \
      --min-timepoints 2 \
      --overwrite

# Stem sequences
python scripts/extract_stem_sequences.py --dataset-path datasets/Pheno4D/processed/v2
```

`configs/point_splits/` holds the 12 plant-level splits, which are identical to GrowFields', and
the 73 organ-level splits derived from them. To regenerate the organ-level splits:

```bash
python scripts/generate_organ_splits.py \
      --dataset-path datasets/Pheno4D/processed/v2 \
      --plant-splits-dir configs/point_splits \
      --output-dir configs/point_splits
```

`eval_metrics.py` reads the ground truth and its normalization from the local dataset. Evaluate
on the same processed copy you trained on. A copy prepared with a different
plant-shape-analysis version can be aligned differently, which shows up as a large CD offset.

### One sequence

`run_train_eval.sh` trains, extracts meshes for every checkpoint (`eval_sphere.py` steps 1
and 2) and computes metrics:

```bash
BASE_RESULTS_DIR=results bash run_train_eval.sh \
      datasets/Pheno4D/processed/v2/organ_sequences/tomato/tomato2_control_plant3_leaf1 ode_solve
```

Results go to `results/pheno4d/<method>/<crop>/<sequence>/<timestamp>_<run>/`, with metrics in
`metrics/metrics_epoch_<epoch>.json`.

### Tables 1 and 2

Run every test sequence for each method, then aggregate at epoch 15,000. `--n_runs 5` seeds
the runs 3408–3412, so it gives a fresh five-seed estimate rather than the published runs:

```bash
for METHOD in TPSR NVFI 4DSDF ode_solve; do
    BASE_RESULTS_DIR=results bash run_crop_sequences.sh all $METHOD plant test --n_runs 5  # Table 1
    BASE_RESULTS_DIR=results bash run_crop_sequences.sh all $METHOD leaf test --n_runs 5   # Table 2
done

# Table 1
python metrics/aggregate_results.py --split test --filter plants --export_csv --experiment_type regular
python scripts/create_latex_table.py --csv output/tables/results_test_plants_epoch15000.csv \
      --per_sequence --output output/tables/results_test_plants_per-plant_epoch15000.tex

# Table 2
python metrics/aggregate_results.py --split test --filter organs --export_csv --experiment_type regular
python scripts/create_latex_table.py --csv output/tables/results_test_organs_epoch15000.csv \
      --per_sequence --output output/tables/results_test_organs_per-plant_epoch15000.tex
```

`leaf` covers all 49 test organ sequences, stems included. The paper evaluates 45: the four
listed under `EXCLUDE` in `launch_test_jobs.sh` are too short and were never run. Skip them
too, by passing the sequences explicitly or deleting their result directories before
aggregating.

Helper scripts:

- `scripts/batch_eval_metrics.py` recomputes metrics for existing runs, e.g.
  `--method ode_solve --filter organs --test_only --experiment_type regular`.
- `scripts/reeval_latest.sh <methods...>` regenerates meshes and metrics for the latest run of
  each sequence, without retraining.
- `check_missing_results.py <method> all plant test --n_runs 5` lists runs that are missing
  or incomplete.
- `scripts/verify_results_counts.py` checks the two CSVs above: 8 plant sequences and 192 leaf
  tips, and 45 organ sequences and 189 leaf tips, for every method and run.

On a SLURM cluster, `launch_test_jobs.sh` submits one job per method and sequence through
`run_euler_24GB.sh` (24 GB GPU, the default) or `run_euler.sh` (80 GB). Both are written for ETH's Euler cluster, so adapt
their module and account lines before using them.

## Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
    @InProceedings{CanFields2025,
    title={CanFields: Consolidating Diffeomorphic Flows for Non-Rigid 4D Interpolation from Arbitrary-Length Sequences},
    author={Wang, Miaowei and Li, Changjian and Vaxman, Amir},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2025}}
   ```

If you use the TrackPlant3D baselines, please also cite GrowFields; the BibTeX entry is in its
[README](https://github.com/Joaquin-Gajardo/growfields#citation).
