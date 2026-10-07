# SnapMMD: forecasting stochastic dynamics beyond the Schrödinger bridge's end

Code and data to reproduce the paper.

| Folder | Content |
|---|---|
| `package/` | the `snapMMD` Python package (`pip install ./package`) |
| `data/` | the snapshot data of the six tasks |
| `reproduction/` | everything needed to reproduce the paper's tables and figures: checkpoints, baseline outputs, evaluation code, and the training scripts of every model and baseline (`reproduction/fit/`); start with `reproduction/README.md` |

Quick start:

```bash
conda env create -f reproduction/environment.yml && conda activate snapmmd-reproduction
cd reproduction/code && python simulate.py && python sanity.py && python score.py && python tables.py && python figures.py
```
