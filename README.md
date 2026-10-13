> **About this fork.** This fork of [wangmiaowei/CanFields](https://github.com/wangmiaowei/CanFields)
> runs the NDF, NVFi, DSR and CanFields baselines of
> [GrowFields](https://github.com/Joaquin-Gajardo/growfields) (ECCV 2026) on the TrackPlant3D
> plant growth dataset. The methods are the original authors'; the fork adds a TrackPlant3D
> loader, the evaluation splits and metrics, the scripts to run them, and fixes to two
> evaluation bugs. See [GrowFields baselines on TrackPlant3D](#growfields-baselines-on-trackplant3d)
> below.

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
git clone https://github.com/Joaquin-Gajardo/CanFields-TrackPlant3D.git
cd CanFields-TrackPlant3D

# Create conda environment with Python 3.9
conda create -n canfields python=3.9 -y
conda activate canfields

# Get gcc and g++ compilers for Pytorch3D
conda install -c conda-forge gcc gxx

# Install pytorch, pytorch3d, torchmeta, plant-shape-analysis and other dependencies
pip install -r requirements.txt

# Alternatively, install the same dependencies step by step
# # PyTorch 1.13.0, the PyPI build (CUDA 11.7) the GrowFields runs used
# pip install torch==1.13.0 torchvision==0.14.0

# # PyTorch3D 0.7.2 wheel (built for CUDA 11.6, runs on the build above). See https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
# pip install pytorch3d@https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1130/pytorch3d-0.7.2-cp39-cp39-linux_x86_64.whl

# # torchmeta 1.8.0 from a fork (the original requires torch<1.10.0 and torchvision<0.11.0)
# pip install git+https://github.com/Joaquin-Gajardo/pytorch-meta.git@1.8.0-torchvision-fix

# # Remaining dependencies
# pip install GPUtil==1.4.0 numpy==1.22.4 plyfile==1.1 point_cloud_utils==0.30.4 pynvml==11.5.0 pytest==8.1.1 scikit_learn==1.3.2 scipy==1.7.3 scikit-image==0.19.3 torchdiffeq==0.2.3 wandb==0.16.3 trimesh==4.9.0 open3d==0.19.0

# # TrackPlant3D loader, same commit as GrowFields
# pip install git+https://github.com/Joaquin-Gajardo/plant-shape-analysis.git@96d9eab34083efe91fe3c01ce2b2a61248b2d7e8
```

<details>
<summary>Upstream examples: quick evaluation and fitting from raw scans (Deforming4D)</summary>

### Quick evaluation

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

### Fitting from raw scans

To train CanFields using sampled points from raw scans, execute:

```bash
# Train CanFields
python train_debugV2.py -a 0.05,0.05,0.01 -o 50002_one_leg_loose_1 -d animals -u True -m False -c 0,15,1 -ev 7000
```

</details>

---

## GrowFields baselines on TrackPlant3D

These are the commands behind the NDF, NVFi, DSR and CanFields rows of Tables 1 (full plant)
and 2 (per organ) in the GrowFields paper. The baselines were not run in the interpolation,
extrapolation or robustness experiments.

The published rows were trained with the code of commit [`a865e96`](https://github.com/Joaquin-Gajardo/CanFields-TrackPlant3D/commit/a865e966c4c86d02dc63772f097b518a2bfa3500),
all at seed 3407. Two fixes below came later; each says how it changes those numbers, and
caveat 8 of GrowFields' [`REPRODUCIBILITY.md`](https://github.com/Joaquin-Gajardo/growfields/blob/main/REPRODUCIBILITY.md#known-inconsistenciescaveats) has the details.

### Methods

| Paper name | `--method` | Consolidator (`-u`) |
| --- | --- | --- |
| NDF | `TPSR` | False |
| NVFi | `NVFI` | False |
| DSR | `4DSDF` | False |
| CanFields | `ode_solve` | True |

`run_train_eval.sh` sets `-u` from the method. `OF4D` is also implemented but was not used in
the paper: it ran out of GPU memory on TrackPlant3D. DSR has no EPE in the paper: it models each frame independently, so it gives no
point correspondences to track.

### What differs from upstream

- **A regularizer fix for NDF and NVFi** (`networks/other_volocities.py`). The velocity
  regularizer expects its Jacobian as `[dt, dx, dy, dz]`. NVFi's was passed with the time
  column in the wrong place, and NDF's dynamics are autonomous, so a zero time column is now
  prepended. Included in the published runs.
- **CanFields' random frequencies are saved in checkpoints** (`networks/multi_velocity_debug.py`).
  Upstream draws `VelocityLayer`'s Fourier frequencies at random on every model build without
  saving them, so evaluation used untrained ones. Checkpoints without them, including all
  published runs, get the seed-3407 training draw rebuilt on load; set
  `CANFIELDS_LEGACY_SEED` for one trained with another seed. The published numbers predate
  this fix: re-evaluated with the training frequencies, CanFields' Table 1 means go from CD
  615.7 and EPE 44.4 to 130.1 and 37.9 (see
  [caveat 8](https://github.com/Joaquin-Gajardo/growfields/blob/main/REPRODUCIBILITY.md#known-inconsistenciescaveats)).
- **Point splits follow the file's point order** (`datasets/data_pheno4d.py`, `eval_metrics.py`).
  Deduplication used to reorder each frame before the split indices were applied, so the
  published runs trained on a random half of the points that overlaps the test split. They
  predate this fix and were not retrained; retraining DSR with it moves CD by under 1%.
- **`-ev 7000`, as upstream and in the published runs.** `-ev` is the iteration at which the
  velocity regularizers switch on, not the number of epochs (training runs for 15,000).
- **Seeds.** The published runs all used seed 3407, so their five runs per sequence are
  identical for NDF, NVFi and DSR, and differ for CanFields only through PyTorch3D's
  nondeterministic KNN. The code now seeds each run with `3407 + run_idx`; `--n_runs 1` (the
  default) uses 3407. Prefix any command with `SEED=3407` to use the published seed for every
  run.
- **TrackPlant3D support**: `datasets/data_pheno4d.py`, `datasets/prepare_pheno4d.py`, the
  train/test point splits in `configs/point_splits/`, and the metrics in `eval_metrics.py`
  (Chamfer distance on held-out surface points, and leaf-tip EPE matched by organ label).

### Data

`prepare_pheno4d.py` downloads TrackPlant3D v2 through
[plant-shape-analysis](https://github.com/Joaquin-Gajardo/plant-shape-analysis) (installed by
`requirements.txt`) and writes plant and organ sequences under
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
bash run_train_eval.sh \
      datasets/Pheno4D/processed/v2/organ_sequences/tomato/tomato2_control_plant3_leaf1 ode_solve
```

Results go to `results/pheno4d/<method>/<crop>/<sequence>/<timestamp>_<run>/`, with metrics in
`metrics/metrics_epoch_<epoch>.json`.

### Tables 1 and 2

The paper's rows are 4 methods × 53 test sequences (8 plants, 45 organs) × 5 runs. One run takes
13 minutes (DSR) to about 3 hours (NVFi) on a GPU, so the full set is roughly 2,000 GPU-hours:
run it on a cluster. `--n_runs 5` seeds the runs 3408–3412, a fresh five-seed estimate; prefix
the commands with `SEED=3407` to repeat the published setup, five runs at seed 3407.

On a SLURM cluster, `launch_test_jobs.sh` submits one job per method and sequence (212 in all),
each running its five runs. DSR and every organ sequence fit in 24 GB; full plants of NDF, NVFi
and CanFields run out of memory there after epoch 5000, so the published ones used 80 GB:

```bash
bash launch_test_jobs.sh --dry-run                     # preview all jobs
bash launch_test_jobs.sh --type leaf                   # organ sequences, all methods (24 GB)
bash launch_test_jobs.sh --methods "4DSDF" --type plant  # DSR full plants (24 GB)
SBATCH_GRES=gpumem:80g bash launch_test_jobs.sh --methods "TPSR NVFI ode_solve" --type plant
```

The jobs go through `run_euler.sh`, which is written for ETH's Euler cluster, so adapt its
module lines first. Activate the `canfields` environment before submitting, since the jobs
inherit it, and set `BASE_RESULTS_DIR` to write somewhere other than `results/`, such as
cluster scratch. Without a cluster, `run_crop_sequences.sh` runs one method's sequences one
after another, e.g. `bash run_crop_sequences.sh all 4DSDF plant test --n_runs 5`.

Both skip sequences with fewer than 3 frames, as the paper does. This leaves 45 of the 49 test
organ sequences (stems included); the four dropped organs have 2 frames each. Prefix
`MIN_TIMESTEPS=2` to run all 49.

Then aggregate at epoch 15,000:

```bash
# Table 1
python metrics/aggregate_results.py --split test --filter plants --export_csv --experiment_type regular
python scripts/create_latex_table.py --csv output/tables/results_test_plants_epoch15000.csv \
      --per_sequence --output output/tables/results_test_plants_per-plant_epoch15000.tex

# Table 2
python metrics/aggregate_results.py --split test --filter organs --export_csv --experiment_type regular
python scripts/create_latex_table.py --csv output/tables/results_test_organs_epoch15000.csv \
      --per_sequence --output output/tables/results_test_organs_per-plant_epoch15000.tex
```

Helper scripts:

- `scripts/batch_eval_metrics.py` recomputes metrics for existing runs, e.g.
  `--method ode_solve --filter organs --test_only --experiment_type regular`.
- `scripts/reeval_latest.sh <methods...>` regenerates meshes and metrics for the latest run of
  each sequence, without retraining.
- `check_missing_results.py <method> all plant test --n_runs 5` lists runs that are missing
  or incomplete.
- `scripts/verify_results_counts.py` checks the two CSVs above: 8 plant sequences and 192 leaf
  tips, and 45 organ sequences and 189 leaf tips, for every method and run.

## Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
    @InProceedings{CanFields2025,
    title={CanFields: Consolidating Diffeomorphic Flows for Non-Rigid 4D Interpolation from Arbitrary-Length Sequences},
    author={Wang, Miaowei and Li, Changjian and Vaxman, Amir},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2025}}
   ```

If you use find this fork useful, please also cite GrowFields; the BibTeX entry is in its
[README](https://github.com/Joaquin-Gajardo/growfields#citation).
