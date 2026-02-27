# CanFields Data Loading Documentation

This directory contains comprehensive documentation explaining how CanFields loads, processes, and trains on plant point cloud data.

## Quick Start Guide

### For a Quick Overview
Start with **`KEY_FILES_REFERENCE.md`** (5 min read)
- File location table
- Function signatures
- Example training command
- Directory structures

### For Detailed Understanding  
Read **`DATA_LOADING_GUIDE.md`** (15 min read)
- Complete data loading pipeline
- Dataset class documentation
- Training loop structure
- Configuration explanation
- Data format specifications

### For Implementation Deep Dive
Consult **`CANFIELDS_DATA_LOADING_INDEX.md`** (20+ min reference)
- Line-by-line code flow
- All processing steps
- Complete examples
- Edge cases and special handling

## Document Contents Summary

### DATA_LOADING_GUIDE.md
1. Overview of CanFields
2. Main training scripts (train_debugV2.py)
3. Dataset classes (DeformSamples for Pheno4D and others)
4. Data format expectations (input/output)
5. Training loop structure and Trainer classes
6. Configuration files (specs.json, data_splits.json)
7. Implementation details (normalization, temporal encoding)
8. Running training with example commands

### KEY_FILES_REFERENCE.md
Quick lookup tables for:
- Core training files
- Dataset loaders
- Configuration files
- Network architecture files
- Loss functions
- Metrics and evaluation
- Data processing utilities
- Directory structures
- Key classes and methods

### CANFIELDS_DATA_LOADING_INDEX.md
Complete implementation reference:
1. Entry point: train_debugV2.py
2. Dataset loading pipeline (load_dataset function)
3. Data representation and batch format
4. Data splitting strategy (plant-level and point-level)
5. Training loop structure (main loop and forward epoch)
6. Configuration files in detail
7. Implementation details (deduplication, filtering, normalization)
8. Dataset-specific implementations
9. Metrics computation
10. Complete file location summary
11. Key takeaways

## Key Findings

### Data Loading Path
```
train_debugV2.py
  ↓ (calls load_dataset)
multi_tools.py: load_dataset()
  ↓ (selects dataset)
datasets/data_pheno4d.py: DeformSamples
  ↓ (reads and processes)
training loop
```

### Data Format
- **Input:** PLY files with point positions and optional normals
- **Processing:** Deduplication, filtering, normalization, temporal encoding
- **Output:** {pts: normalized [N,3], nms: normals [N,3], normed_time: scalar}

### Normalization
```python
mean = point_cloud.mean(0)
scale = 1.0 / max(|point_cloud - mean|)
normalized = (point_cloud - mean) * scale
```

### Data Splits
- **Plant-level:** Val (4 plants, 15 sequences) vs Test (8 plants, 29 sequences)
- **Point-level:** 50/50 train/test per timepoint for evaluation
- **Strategy:** Avoid leakage in leaf stitching

## File Locations

All documented files are in this repository:

### Documentation Files (in root)
- `DATA_LOADING_GUIDE.md` - Main detailed guide
- `KEY_FILES_REFERENCE.md` - Quick reference
- `CANFIELDS_DATA_LOADING_INDEX.md` - Implementation reference
- `DATA_DOCUMENTATION_README.md` - This file

### Source Code Files
- `train_debugV2.py` - Main training script (entry point)
- `multi_tools.py` - Trainer classes, dataset loading
- `datasets/data_pheno4d.py` - Pheno4D dataset loader
- `datasets/data_wmw.py` - Animals dataset loader
- `configs/specs.json` - Training specifications
- `configs/data_splits.json` - Dataset splits
- `configs/point_splits/` - Point-level splits

## How to Use This Documentation

### Scenario 1: I want to understand the overall flow
1. Read KEY_FILES_REFERENCE.md section "Key Data Flow"
2. Then read DATA_LOADING_GUIDE.md sections 1-3
3. Look at the example command in KEY_FILES_REFERENCE.md

### Scenario 2: I want to add a new dataset
1. Read KEY_FILES_REFERENCE.md to see dataset structure
2. Study datasets/data_pheno4d.py as a template
3. Reference CANFIELDS_DATA_LOADING_INDEX.md section 8 for other implementations
4. Follow the DeformSamples class pattern

### Scenario 3: I want to understand how data is normalized
1. Read DATA_LOADING_GUIDE.md section 4 "Data Format Expectations"
2. Review CANFIELDS_DATA_LOADING_INDEX.md section 3 "Data Representation"
3. Check implementation details section 7 in both guides

### Scenario 4: I want to run a training experiment
1. Check KEY_FILES_REFERENCE.md "Example Training Command"
2. Review DATA_LOADING_GUIDE.md section 8 "Running Training"
3. Understand the --clip parameter in section 2

### Scenario 5: I want to understand the data splits
1. Read DATA_LOADING_GUIDE.md section 3
2. Check configs/data_splits.json file directly
3. Review CANFIELDS_DATA_LOADING_INDEX.md section 4

## Important Classes and Methods

### Trainer (main_tools.py)
- `__init__()` - Initialize with dataset and networks
- `forward_epoch(epoch)` - Main training iteration
- `compute_and_save_metrics(epoch)` - Evaluate and save metrics
- `save_model(epoch, directory)` - Save checkpoints

### DeformSamples (data_pheno4d.py)
- `__init__(filename, start_idx, gap, train_step, use_dense)` - Load dataset
- `__getitem__(idx)` - Get sample with pts, nms, normed_time
- `get_leaf_tips(idx)` - Get leaf tip indices
- `evaluate_train(idx)` - Get all points for mesh generation

## Command Line Arguments Explained

| Argument | Format | Example | Purpose |
|----------|--------|---------|---------|
| `--clip` | "start_idx,gap,stride" | "0,9,1" | Select frames 0 to 9 with stride 1 |
| `--adjust_weights` | "l2,elas,prob" | "0.05,0.05,0.01" | Adjustment network loss weights |
| `--dataset_name` | name | "pheno4d" | Choose dataset type |
| `--object_name` | path | "datasets/Pheno4D/.../plant1" | Specific sequence path |
| `--adjust_use` | True/False | "True" | Enable adjustment network |
| `--epoch_velo` | number | "7000" | Start velocity constraints at epoch |
| `--methods` | method | "ode_solve" | Choose method variant |

## Key Statistics

### Dataset Sizes
- **Pheno4D Val:** 4 plants, 15 sequences, ~9-16 frames per sequence
- **Pheno4D Test:** 8 plants, 29 sequences, various lengths
- **Total:** 44 plant leaf sequences, ~12 plants

### Model Configuration
- **Latent code dimension:** 128
- **Training epochs:** 15001
- **Samples per scene:** 5000
- **Checkpoint frequency:** Every 5000 epochs

### Normalization Details
- **Point deduplication tolerance:** 1e-15
- **Normal filtering threshold:** 1e-5
- **Temporal encoding range:** [0, 1]
- **Bounding box expansion:** 1.6x or 1.7x

## Troubleshooting

### Understanding Data Loading Issues
- See CANFIELDS_DATA_LOADING_INDEX.md section 2 "Dataset Loading Pipeline"
- Check actual PLY file locations match expected structure
- Verify timestamps extracted from filenames (D00, D01 format)

### Understanding Training Loops
- See DATA_LOADING_GUIDE.md section 5
- Trace through CANFIELDS_DATA_LOADING_INDEX.md section 5 "Training Loop Structure"

### Understanding Data Normalization
- See DATA_LOADING_GUIDE.md section 4
- Review implementation in datasets/data_pheno4d.py lines 180-192

## References

All code file references are to the absolute path:
`/mnt/Data/jgajardo/code/CanFields/`

Configuration and data files referenced:
- Specs: `configs/specs.json`
- Splits: `configs/data_splits.json`
- Point splits: `configs/point_splits/*.json`
- Datasets: `datasets/Pheno4D/processed/v2/organ_sequences/`

## Version Information

- **Project:** CanFields: Consolidating 4D Dynamic Shapes (ICCV 2025)
- **Documentation Created:** November 2025
- **Python Version:** 3.9
- **Key Dependencies:** PyTorch, Open3D, pytorch3d, point_cloud_utils

---

For more information, start with **KEY_FILES_REFERENCE.md** for a quick overview, 
then proceed to **DATA_LOADING_GUIDE.md** for detailed understanding.
