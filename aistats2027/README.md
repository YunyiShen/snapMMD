# Reproducing the AISTATS 2027 paper

This folder reproduces every table and the computed figures of the paper *Forecasting Stochastic Dynamics Beyond the Schrödinger Bridge's End* from the shipped checkpoints and outputs, and contains the scripts that trained every model. It uses the `snapMMD` package (`../package`) and the released data (`../data`) of this repository; the four particle-simulation baselines are in `../forecasting_baselines`.

## Quick start (laptop, about 1–2 hours, no training)

```bash
conda env create -f environment.yml && conda activate snapmmd-aistats2027
cd code
python simulate.py     # forecasts, paths at the validation times and R^2 from the checkpoints -> generated/
python sanity.py       # persistence and OT-midpoint baselines -> generated/
python score.py        # MMD^2 (snapshot bandwidth and the bandwidth sweep) and EMD for every method -> results/scores.csv
python tables.py       # recomputes every cell of every results table and compares it with the paper -> results/check_report.txt
python figures.py      # forecast, vector-field and interpolation-metric figures -> results/figures/figs/ (needs LaTeX)
```

`results/check_report.txt` lists every table cell whose recomputed value differs from the printed one at the printed precision; `results/recomputed_tables.csv` has all of them.

## What is where

| Folder | Content |
|---|---|
| `checkpoints/snapmmd/<task>/` | SnapMMD fits, 10 seeds per task (Tables 1, 3, App. D) |
| `checkpoints/snapmmd_mrna_only/ReprProtein/` | SnapMMD with the mRNA-only model on the mRNA–protein data (App. D.8) |
| `checkpoints/fixed_volatility/<task>/`, `checkpoints/fully_neural/<task>/` | ablations (Sec. 4.6, App. E) |
| `checkpoints/schrodinger_bridge/{SBIRR,SBforward}/<task>/` | fitted references of SBIRR and SB-forward (pickled objects; need the SBIRR package, see below) |
| `checkpoints/flow_matching/<task>/` | OT-CFM checkpoints |
| `outputs/<method>/<task>/seed_<s>.npz` | baselines: `forecast` (final cloud) and/or `interp` (the path at the validation times only) |
| `outputs/vector_fields/` | vector-field grids of seed 42 (true field, SnapMMD, SBIRR-ref, SB-forward) for the App. D figures |
| `code/` | simulation, metric, scoring, tables, figures; `code/models/` has the model classes |
| `fit/` | the training scripts of every model (see below) |
| `paper_tables/` | the LaTeX sources of the paper's results tables, used by `tables.py` for the comparison |

Tasks: `LV` (Lotka–Volterra), `ReprParam` / `ReprSemiparam` (repressilator, parametric / semiparametric model family), `ReprProtein` (mRNA–protein repressilator, mRNA observed), `GoM` (Gulf of Mexico), `PBMC` (T cell-mediated immune activation). Methods without a model family share their fits across `ReprParam` and `ReprSemiparam`. Seeds: 1, 2, 3, 4, 5, 40, 41, 42, 43, 44.

**Protocol.** Every simulation starts from the observed first snapshot (each point repeated 5 times in training; unobserved protein coordinates at 0). Forecasts and paths of SnapMMD and its ablations are simulated by `code/simulate.py` from the checkpoints (seeds: seed for the forecast, seed + 1000 for the path, seed + 2000 for R^2), so a reviewer can regenerate and inspect them. Paths are read at the validation times with the rule of `common.midpoints` (middle of each of n_val equal segments of the simulated path), for our method and the baselines alike.

**Metric.** Squared MMD (V-statistic) with one RBF kernel whose bandwidth is the median squared distance within the held-out snapshot (`common.mmd2`, key `snap`); EMD is the exact 2-Wasserstein distance (`common.emd`, POT).

## Training from scratch (`fit/`, cluster)

| Folder | Paper result | Command (run inside the folder; create `models/`, `forecasts/`, `interpolation/` first) |
|---|---|---|
| `fit/snapmmd/classic` | LV, ReprParam | `python classic_sde.py <id>` (id = task index × 10 + seed index) |
| `fit/snapmmd/mlp` | ReprSemiparam | `python MLP.py <id>` |
| `fit/snapmmd/missingobs` | ReprProtein | `python missing_data.py <id>` |
| `fit/snapmmd/realdata` | GoM; PBMC | `python realdata.py <id>` (GoM); `python pbmc42.py --id <seed index> --epochs_override 6000 --lr_override 0.001 --gradclip 1.0` (PBMC) |
| `fit/snapmmd_mrna_only` | App. D.8 | `python fit_mrna_only.py <seed>` |
| `fit/fixed_volatility/*` | App. E.1 | as for `fit/snapmmd` (model classes with the volatility fixed at 0.1); PBMC with the settings of the PBMC fit |
| `fit/fully_neural` | App. E.2 | `python MLP.py --id <id> --gradclip 1.0`; PBMC: `python pbmc42.py` with the PBMC settings and a fully neural drift |
| `fit/baselines/schrodinger_bridge/*` | SBIRR, SBIRR-ref, SB-forward | `python baseline.py <id> <n_tasks> <task>` (needs the SBIRR package) |
| `fit/baselines/flow_matching` | OT-CFM, SB-CFM, SF2M | `python <task>.py --seed <seed>` (needs `torchcfm`) |
| `fit/baselines/dmsb` | DMSB | `python run_DMSB.py <id> <n_tasks> <task>`, then `python extract_dmsb.py` (needs the DMSB code of Chen et al., 2023) |
| `../forecasting_baselines` | PRESCIENT, PI-SDE, scNODE, JKOnet* | see its README |

## Notes

- **What is not shipped.** Full simulated paths of the baselines (about 4 GB) are not included: `outputs/` keeps each baseline's path at the validation times only, which is all the metric uses. The trajectory figures of the appendix (`*_interpolationing.png`, the PBMC progression and interpolation grids) need the full paths and are therefore not regenerated by `figures.py`. The SBIRR interpolation on PBMC uses 10,000 particles and is stored in single precision.
- **SBIRR package.** The SB references in `checkpoints/schrodinger_bridge` are pickled objects of the SBIRR code (the code release of Shen et al., 2025, folder `package/`). Their forecasts propagated from the last snapshot (App. D.3) are shipped in `outputs/SBIRR-ref (last snapshot)` and `outputs/SB-forward (last snapshot)`.
- **PI-SDE licence.** The PI-SDE repository has no licence, so its code is not redistributed here; `../forecasting_baselines/fetch_pisde.sh` downloads it at a fixed commit and applies our patch.
- **Diverged runs.** Two of the ten fully neural ReprSemiparam fits (seeds 2 and 43) diverge during training (their checkpoints contain non-finite parameters); they are left out of App. E.2, as stated there.
