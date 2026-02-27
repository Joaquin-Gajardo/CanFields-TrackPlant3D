# Plan: Add Per-Sequence LaTeX Table to `scripts/create_latex_table.py`

## Context

The paper's LaTeX table currently shows results aggregated across all sequences (one value per method). The DOMA-release repo updated its table to show per-sequence columns with CD and EPE metrics. We need the same format here for the baselines table. The key difference is that CanFields CSVs contain **multiple methods** in one file and data is **already aggregated** per sequence (no timepoint/run level).

## Target Table Structure

```
             | MaC2       | MaC3       | SoC2       | ... | Avg
Method       | CD    EPE  | CD    EPE  | CD    EPE  | ... | CD    EPE
-------------+------------+------------+------------+-----+-----------
4DSDF        | 13.2  45.3 | 28.1  67.2 | ...        | ... | ...  ...
NVFI         | ...        | ...        | ...        | ... | ...  ...
```

- 2 metrics per sequence: **CD** (`mean_cd`) and **EPE** (`mean_epe`)
- Sequence columns use abbreviations (MaC2, SoH2, TbC1, etc.)
- **Avg** column at the end: mean across sequences
- For **organs CSV**: organ sequences (e.g. `maize_control_plant2_leaf1`) are grouped by parent plant → columns are still the 8 parent plants, values are averaged across organs within each parent

## Changes to `scripts/create_latex_table.py`

### 1. Add `abbreviate_sequence(seq_name)` function
- Ported from DOMA-release
- Maps `maize_control_plant2` → `MaC2`, `tomato1_heat_plant3` → `Tm1H3`, etc.
- Species: Ma=maize, So=sorghum, Tb=tobacco, Tm1=tomato1, Tm2=tomato2
- Condition: C=control, H=heat/highlight, S=shade

### 2. Add `get_parent_sequence(seq_name)` function
- Extracts parent plant from organ sequence names
- `maize_control_plant2_leaf1` → `maize_control_plant2`
- Rule: keep first 3 underscore-separated parts (`species_condition_plantN`)

### 3. Add `detect_organ_sequences(sequences)` function
- Returns True if sequences contain organ suffixes (4+ underscore parts with leaf/stem)
- Used to auto-detect whether grouping by parent plant is needed

### 4. Add `compute_per_sequence_stats(df, split)` function
- For **plants CSV**: group by (method, sequence) → use `mean_cd` and `mean_epe` directly
- For **organs CSV**: add `parent_sequence` column, group by (method, parent_sequence) → average `mean_cd` and `mean_epe` across organs
- Returns 3-tuple: `(dict {method: OrderedDict({sequence: {cd: val, epe: val}})}, sorted_sequences, flat_avgs)`
- `flat_avgs`: `{method: {cd: val, epe: val}}` — flat mean across all individual sequences (not parent-level means), ensuring Avg column matches the default aggregated table
- Sequences are sorted consistently across all methods

### 5. Add `generate_per_sequence_header(sequences)` function
- `\begin{table*}[htbp]` with `\resizebox{\textwidth}{!}{...}`
- Multicolumn headers with abbreviations + **Avg** column
- Metric sub-headers: `CD $\downarrow$` and `EPE $\downarrow$`
- Caption with metric definitions

### 6. Add `generate_per_sequence_footer()` function
- Closes tabular, resizebox, table* environments

### 7. Add `format_per_sequence_row(method, seq_stats, sequences, flat_avg=None)` function
- One value per (sequence, metric) cell — no ±std (baselines are single-run)
- 1 decimal place for both CD and EPE (values are large: CD up to ~36000, EPE up to ~200)
- `--` for missing sequences (e.g., OF4D has only 6/8 plant sequences)
- Avg column: uses pre-computed `flat_avg` (flat mean across all individual organ/plant sequences) to match the default aggregated table; falls back to mean of column values if not provided

### 8. Add CLI flag `--per_sequence`
- When set, uses the new per-sequence table generation path
- Works with existing `--methods`, `--output`, `--split` flags

### 9. Wire up in `main()`
- When `--per_sequence`: detect organs vs plants, compute per-sequence stats, generate full table with header + all method rows + footer
- Existing default behavior unchanged

## Files Modified
- `scripts/create_latex_table.py` — single file, all changes

## Verification
```bash
# Plants per-sequence table
python scripts/create_latex_table.py --csv output/tables/results_test_plants_epoch15000.csv --per_sequence

# Organs per-sequence table (auto-groups by parent plant)
python scripts/create_latex_table.py --csv output/tables/results_test_organs_epoch15000.csv --per_sequence

# With specific methods and output file
python scripts/create_latex_table.py --csv output/tables/results_test_plants_epoch15000.csv --per_sequence --methods 4DSDF NVFI ode_solve --output table.tex

# Verify: 8 sequence columns + Avg, -- for missing data, 1 decimal place
```