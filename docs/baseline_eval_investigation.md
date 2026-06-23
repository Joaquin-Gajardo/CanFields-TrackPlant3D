# Baseline evaluation investigation — CanFields on Pheno4D

_Last updated: 2026-06-23_

This note records what we investigated about the "extremely bad" baseline numbers
(TPSR, 4DSDF, NVFI, ode_solve) on the Pheno4D test split, what the root cause(s)
turned out to be, and the open items for the velocity-network regularizers.

---

## 1. TL;DR

The catastrophic Chamfer-Distance (CD) numbers in the original results table were
**not** a method/hyperparameter problem. There is essentially **one root cause**, at
*evaluation* time (not training time):

> **GT / dataset version mismatch.** The models were trained on an HPC copy of the
> dataset from **13 Nov**, but the *local* dataset is from **27 Oct**, and the leaf
> alignment in `../plant_shape_analysis` changed *in between*. `eval_metrics.py` reads
> the GT **and** the normalization (`mean`/`scale`) from the *local* dataset, so it
> was comparing HPC-trained predictions against a differently-aligned GT.

This single mismatch produced **two visible symptoms**, which earlier looked like two
separate bugs:

1. **A gross ~18 mm translation offset on the stored deformed meshes.** Those meshes
   were denormalized in the **13-Nov training frame** but scored against the **27-Oct**
   local GT. Regenerating them *locally* removes the offset — not because the data was
   fixed, but because denormalization and GT then come from the *same* local dataset
   (self-consistent). Verified end-to-end (ode_solve leaf1 CD 543.6 → 4.45, NVFI leaf1
   469.5 → 3.59).

2. **Residual per-organ error** for sequences whose alignment actually changed between
   Oct and Nov. Regenerating makes eval self-consistent in the 27-Oct frame, but that
   is still not the frame the model was trained in, so a gap remains. This is the
   elevated-CD tail still visible after reeval.

The proper fix for **both** symptoms is to align the local dataset to the training
version (see §2) — then the gross offset never appears and the residual error closes.

There is also a smaller, **genuine** category of failures (degenerate canonical
meshes on small organs, shape over-extension at temporal extremes) that no amount
of re-evaluation or re-alignment fixes — those need retraining or are honest failure
cases.

---

## 2. Root cause in detail: GT version mismatch (action required)

### Why it matters

`eval_metrics.py` does **not** use any normalization stored from training. It
recomputes everything from whatever local dataset it finds:

- `eval_metrics.py:569-570` — `scale = sdf_dataset.scale`, `mean = sdf_dataset.mean`
  (recomputed from the local dataset).
- `eval_metrics.py:283-333` — GT point clouds loaded from the local
  `sparse_pc/` folder.
- Predictions are denormalized with the local `mean`/`scale`
  (`eval_metrics.py:712-713, 725-726`) and then compared to the local GT.

So the local dataset version drives **both** the denormalization frame **and** the
comparison target. If the local GT alignment ≠ the alignment the model was trained
on, you get a translation/scale offset *and* a wrong comparison target — exactly the
symptom we saw.

### Confirmation that it's improving

`output/tables/results_test_organs_epoch15000.csv` (regenerated) is already better
than `output/tables/results_test_organs_epoch15000_old2.csv` (stale-mesh era),
consistent with this explanation.

### What to do — **yes, regenerate the local GT**

You should re-run `datasets/prepare_pheno4d.py` locally so the local dataset matches
the alignment used during HPC training. **Important caveats:**

- `prepare_pheno4d.py` is deterministic given its input, but its input is the
  **aligned point clouds produced by `../plant_shape_analysis`**. Regenerating only
  reproduces the 13 Nov dataset if that alignment source is at the **same state** as
  it was for the 13 Nov HPC run.
- If `../plant_shape_analysis` changed *again* after 13 Nov, regenerating *now* would
  produce a **third** version that matches neither the HPC runs nor the old table.
  Check the `plant_shape_analysis` git log and pin/checkout the commit that was
  current on 13 Nov before regenerating.
- **Safest option:** copy the exact processed dataset
  (`datasets/Pheno4D/processed/v2/`) from the HPC cluster. That guarantees a
  byte-for-byte match with what the models were trained on, with no risk of a
  silent re-alignment. Re-deriving locally is the second-best option.

Either way, the rule is: **the GT used at eval time must be the same alignment the
model was trained on**, because eval re-derives normalization from it.

### After regenerating the dataset, you must re-run `reeval_latest.sh`

Regenerating the local dataset changes `mean`/`scale` and the GT, so all previously
generated meshes/metrics become stale:

- `eval_sphere.py -s 2` warps **and denormalizes** using `mean`/`scale` derived from
  the dataset → deformed meshes must be regenerated.
- `eval_metrics.py` loads GT and normalization from the dataset → metrics must be
  recomputed.
- (`eval_sphere.py -s 1` canonical mesh is dataset-independent, but `reeval_latest.sh`
  re-runs all three steps with `--overwrite` anyway, so just run it again.)

```bash
CUDA_VISIBLE_DEVICES=1 TEST_ONLY=1 bash scripts/reeval_latest.sh ode_solve 4DSDF NVFI TPSR \
  2>&1 | tee sbatch_log/reeval_test.log
```

---

## 3. The translation-offset symptom (stale deformed meshes)

- The eval pipeline is: `eval_sphere.py -s 1` (canonical mesh) →
  `eval_sphere.py -s 2` (warp template to each frame + denormalize + export) →
  `eval_metrics.py` (sample meshes, compute CD/EPE).
- `eval_sphere.py` **skips** regeneration if PLYs already exist unless `--overwrite`
  is passed. The original bad table reused old (13 Nov) meshes.
- `scripts/reeval_latest.sh` regenerates the latest experiment folder per sequence
  with `--overwrite`, then computes metrics **inline** for just that sequence (it
  calls `eval_metrics.py` directly — no need for `batch_eval_metrics.py`, which would
  recompute everything).

### Current test-split coverage (after reeval)

Target = 8 plants + 49 organs = 57 sequences.

| Method    | Fresh | Stale | No checkpoint folder | Folder but no mesh |
|-----------|------:|------:|---------------------:|-------------------:|
| ode_solve |    53 |     0 |                    4 |                  0 |
| 4DSDF     |    53 |     0 |                    4 |                  0 |
| NVFI      |    53 |     0 |                    4 |                  0 |
| TPSR      |    51 |     0 |                    4 |                  2 |

- 4 organs were never trained for any method:
  `maize_control_plant2_leaf5`, `maize_control_plant3_leaf4`,
  `sorghum_highlight_plant2_leaf5`, `tobacco_control_plant1_leaf6`.
- TPSR special cases:
  - `tobacco_control_plant1_leaf2` — only reached epoch 10000 (no 15000 ckpt);
    re-evaluated at 10000 but CD ≈ 4.2e6 → **genuinely broken run**, needs retraining.
  - `tomato1_shade_plant1_leaf8` — 15000 ckpt fails marching cubes
    ("Surface level must be within volume data range"); evaluated at 5000 → CD ≈ 0.66.

### Aggregation note

`metrics/aggregate_results.py` uses `find_best_available_epoch` (line 60): asking for
`--epoch 15000` falls back to the closest **earlier** epoch when 15000 is absent, so
the two TPSR sequences above (10000 and 5000) are picked up automatically:

```bash
python metrics/aggregate_results.py --epoch 15000 --split test --export_csv --export_latex
```

---

## 4. Velocity-network regularizers — status and possible modifications

### Background (paper vs. code)

- "epoch" in the code == "iteration" in the paper. Total fitting length is controlled
  by `NumEpochs = 15001` in `configs/specs.json` / `configs/others_specs.json`.
- The velocity regularizers (acceleration, Killing energy, Dirichlet energy) are gated
  off until iteration `epoch_velo`, then ramped up. Paper: off for the first 7,000
  iters, ramped over the next ~5,000, held for the last ~3,000.
- `run_train_eval.sh` now passes `-ev 7000` (`VELO_EPOCH=7000`).
- Dynamic-consolidator (adaptor) regularizers (`-a 0.05,0.05,0.01`) are active from the
  start and only apply to `ode_solve` (adjust_use=True).

### The gate and ramp (verified line refs)

- Gate (both trainers): `if (self.epoch_velo != -1) and (epoch > self.epoch_velo)`
  - ode_solve `Trainer`: `multi_tools.py:739, 782, 792, 816`
  - baselines `Trainer_woadjust`: `multi_tools.py:1239, 1265, 1275, 1291`
- Ramp denominators differ between the two trainers:
  - ode_solve: `20000 - epoch_velo` (`multi_tools.py:825, 828, 831`)
  - baselines: `num_epochs - epoch_velo` = 15001 − 7000 = 8001
    (`multi_tools.py:1296, 1299, 1302`)
- The regularizer itself, `comput_kL_L2` (`multi_tools.py:649`), expects a **4-column
  Jacobian** ordered `[∂/∂t, ∂/∂x, ∂/∂y, ∂/∂z]`:
  - acceleration: `this_grad[:, :, 0].norm(...)`  (line 657)
  - Killing: symmetrizes `this_grad[:, :, 1:]`     (lines 658-661)
  - Dirichlet: `this_grad[:,:,1]**2 + [:,:,2]**2 + [:,:,3]**2`  (lines 662-666)

### Per-method correctness of the velocity Jacobian

| Method    | Velocity net (file)                       | Jacobian columns fed to `comput_kL_L2` | Status |
|-----------|-------------------------------------------|----------------------------------------|--------|
| ode_solve | native velocity field                     | `[∂t,∂x,∂y,∂z]` (4 cols), correct       | OK     |
| OF4D      | `networks/of4d_velo.py` (p=[t-0.5,x,y,z]) | `[N,3,4]`, time-first, correct          | OK (but excluded from runs) |
| NVFI      | `VelocityAABBSur.forward` (`other_volocities.py:169`) | **scrambled** — line 180 does `cat(derivs[:,1:], derivs[:,:1])` → `[∂y,∂z,∂t,∂x]` | **BUG if reg enabled** |
| TPSR      | `other_volocities.py` (3D state)          | grads are `[N,3,3]` (no time column)    | **CRASH if reg enabled** (indexing col 3) |
| 4DSDF     | `id_flag=True` (no warp)                  | n/a — no velocity field to regularize   | n/a    |

So today, turning the velocity regularizer **on** for the baselines is only safe for
ode_solve (and OF4D). NVFI would silently penalize the wrong derivatives, and TPSR
would index a non-existent 4th column.

### Modifications we *could* still make

1. **Fix NVFI ordering** (`networks/other_volocities.py:180`): the
   `torch.cat((derivs[:, 1:], derivs[:, :1]), dim=1)` reshuffle puts time **last**.
   To feed `comput_kL_L2` correctly the Jacobian must be `[∂t,∂x,∂y,∂z]` (time first).
   Either remove/replace that reorder so time stays in column 0, or adapt the consumer.
   Only needed if NVFI is retrained with `-ev` enabled.

2. **Fix TPSR Jacobian** (3-column → 4-column): TPSR's state is 3D and `return_grads`
   yields `[N,3,3]` (`other_volocities.py:291-296`), so `comput_kL_L2`'s references to
   column 3 (Dirichlet/Killing) would crash. Would need either a time column added or a
   TPSR-specific regularizer path. Only needed if TPSR is retrained with `-ev` enabled.

3. **Optional ramp-denominator fidelity tweak** (`multi_tools.py:825/828/831` for
   ode_solve): the paper ramps over ~5,000 iters then holds; with `epoch_velo=7000`
   that means reaching full strength by iteration 12,000. The code currently divides by
   `20000 - epoch_velo = 13000`, so it never fully reaches the target weight within the
   15,001-iter budget. Changing `20000` → `12000` would make ode_solve match the paper's
   "ramp over 5k, hold last 3k" schedule. (The baseline trainer divides by
   `num_epochs - epoch_velo`, a different choice; align if desired.)

> These are **only** relevant if you decide to retrain the baselines with the velocity
> regularizers active. They have no effect on the re-evaluation of existing checkpoints.

---

## 5. Quick command reference

Regenerate meshes + metrics for the latest folder of each **test** sequence (4 methods,
excluding OF4D), pinned to GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 TEST_ONLY=1 bash scripts/reeval_latest.sh ode_solve 4DSDF NVFI TPSR \
  2>&1 | tee sbatch_log/reeval_test.log
```

Single sequence at a specific epoch (what `reeval_latest.sh` calls internally):

```bash
python eval_metrics.py -d pheno4d -e 10000 \
  -o datasets/Pheno4D/processed/v2/organ_sequences/tobacco/tobacco_control_plant1_leaf2 \
  --experiment_name <EXP_DIR_NAME> --methods TPSR \
  --point_split_file configs/point_splits/tobacco_control_plant1_leaf2_split.json
```

> Note: `eval_sphere.py` / `eval_metrics.py` respect `CUDA_VISIBLE_DEVICES`.
> `train_debugV2.py` does **not** — it auto-selects the idlest GPU via
> `set_most_idle_gpu_as_visible()` (line 27); edit/bypass that to pin training.

Aggregate the test-split table:

```bash
python metrics/aggregate_results.py --epoch 15000 --split test --export_csv --export_latex
```
