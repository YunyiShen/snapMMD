# Forecasting baselines: PRESCIENT, PI-SDE, scNODE, JKOnet*

The four forecasting methods compared with SnapMMD in the paper (Section 5 and Appendix D.3), fitted on the paper's
snapshot data with the authors' default settings and scored exactly as SnapMMD (MMD² with the snapshot bandwidth,
EMD; rollout from the observed first snapshot). One call fits one (method, variant, task, seed):

```bash
python run_baselines.py --method prescient --variant sd0.5    --task LV   --seed 1   # PRESCIENT, volatility 0.5
python run_baselines.py --method pisde     --variant const0.1 --task PBMC --seed 1   # PI-SDE, volatility 0.1 (or: mlp = learned)
python run_baselines.py --method scnode                        --task GoM  --seed 1   # scNODE
python run_baselines.py --method jkonet    --variant potential --task LV   --seed 1   # JKOnet*, potential only (or: full)
python score.py                                                                        # -> results/scores.csv, results/summary.txt
```

`run_all.sh` runs the 200 PRESCIENT / PI-SDE / scNODE fits of the paper (10 seeds × 5 tasks × 4 columns) and
`run_jkonet.sh` the 90 JKOnet* fits; `results/scores.csv` and `results/summary.txt` are the scores reported in the paper.

## Environments

* PRESCIENT, PI-SDE, scNODE: the `snapmmd` environment of this repository plus `pip install --no-deps prescient geomloss
  torchdiffeq` (versions used: prescient 0.1.0, geomloss 0.3.1, torchdiffeq 0.2.5, torch 2.2.1) and POT for EMD.
* JKOnet*: a separate environment with the authors' pins (python 3.12, jax/jaxlib 0.4.26, flax 0.8.3, optax 0.2.2,
  ott-jax 0.4.6, POT 0.9.3, scikit-learn 1.4.2, torch 2.2.2); `run_jkonet.sh` activates it as `jkonet-star`.

## Vendored code

`vendor/prescient_train` (MIT, see `NOTICE.md`), `vendor/scnode` (MIT, `LICENSE`; `model/`, `optim/` and the one helper
of `benchmark/BenchmarkUtils.py` that the training loop needs) and `vendor/jkonet_star` (MIT, `LICENSE`; `models/`,
`networks/`, `utils/`, `dataset.py`, configs; one constant in `utils/density.py` made adjustable, see `NOTES.md`) are
copied from the authors' repositories at the commits recorded in their folders. PI-SDE's repository carries no licence,
so its model file is not redistributed: `bash fetch_pisde.sh` downloads it at the commit we used and applies the one
import change the driver needs.

## What is the authors' and what is ours

The training loops, architectures, losses, optimisers and hyperparameters are the authors' (PRESCIENT `train/run.py`
and `train_model.py` defaults; PI-SDE `src/train.py` and `src/config_Veres.py` defaults; scNODE `optim/running.py`
defaults; JKOnet* `train.py` and `config.yaml` defaults), reproduced on in-memory arrays instead of their data files.
Departures, each recorded in the fit's log and in Appendix D.3 of the paper: time runs on the data's own grid (PBMC in
hours); PI-SDE is integrated with `torchsde.sdeint` (its own solver is a copy of torchsde's); scNODE uses a linear
encoder/decoder and the `dopri5` solver (its ReLU decoder cannot output the negative GoM coordinates; its Euler option
takes one step per requested time); JKOnet*'s determinant filter on the Gaussian-mixture components is set to 0 (its
absolute threshold removes every component on these data), its full model is not run on PBMC, and its interpolation
trajectory is the straight-line interpolation of its own one-step-per-interval predictor. Every method sees the training
snapshots only and starts from the observed first snapshot.
