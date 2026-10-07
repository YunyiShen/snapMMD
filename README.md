# SnapMMD: forecasting stochastic dynamics beyond the Schrödinger bridge's end

Code and data to reproduce the paper.

| Folder | Content |
|---|---|
| `package/` | the `snapMMD` Python package (`pip install ./package`) |
| `data/` | the snapshot data of the six tasks |
| `aistats2027/` | everything needed to reproduce the paper's tables and figures: checkpoints, baseline outputs, evaluation code, and the training scripts of every model and baseline (`aistats2027/fit/`); start with `aistats2027/README.md` |

Quick start:

```bash
conda env create -f aistats2027/environment.yml && conda activate snapmmd-aistats2027
cd aistats2027/code && python simulate.py && python sanity.py && python score.py && python tables.py && python figures.py
```
