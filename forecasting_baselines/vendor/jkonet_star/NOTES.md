# Vendored JKOnet* code

Source: https://github.com/antonioterpin/jkonet-star, commit in `COMMIT.txt` (MIT licence, `LICENSE`).
Copied unchanged: `models/`, `networks/`, `utils/`, `dataset.py`, `config.yaml`, `config-jkonet-extra.yaml`.

One edit: `utils/density.py`, the determinant threshold `1e-4` in `GaussianMixtureModel.fit` is now the module
constant `DET_THRESHOLD` (default unchanged). `run_baselines.run_jkonet` sets it to 0 for our data; the reason is in
its docstring and in `results/RUNLOG.md`.
