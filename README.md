# Codebase for Oh SnapMMD! Forecasting stochastic dynamics beyond the Schrödinger bridge's end

## Structure

The three core folders in this repository are `experiments`, `data`, and `package`. The `package` folder contains our method, implemented as an installable Python package called `snapMMD`. After installing the package, experiments in `experiments` can be run using files from `data`.

## Environment

We recommend the following procedure to run our code:

1. Create a virtual environment with Python 3.8.16. For example, using Conda, run `conda create -n "snapmmd" python=3.8.16 ipython`, then activate it with `conda activate snapmmd`. Other Python versions may work if compatible versions of PyTorch and TorchSDE are installed, but they have not been tested.
2. Install the dependencies from `requirements.txt` by running `pip install -r requirements.txt`.
3. Install the `snapMMD` package by running `pip install ./package`.

## Experiments

To reproduce our experiments, run the scripts in `experiments`. Each script takes an argument specifying which experiment to run.

There are four subfolders in `experiments`:

- `classic`: Classic SDE experiments. There are two experiments: `LV` for Lotka–Volterra and `Repressilator` for the repressilator. The main script is `classic_sde.py`; for example, `python classic_sde.py LV` runs the LV experiment.
- `missingobs`: Experiments with missing observations. There is one experiment, `Repressilator`, which uses a repressilator with missing protein observations. The main script is `missing_data.py`.
- `mlp`: Semiparametric model experiments. There is one experiment, `Repressilator`, which uses a repressilator with an MLP activation function. The main script is `MLP.py`.
- `realdata`: Real-data experiments. There are two experiments: `GoM` for a vortex in the Gulf of Mexico and `pbmc` for a single-cell dataset. The main script is `realdata.py`.

## Ablations and baselines

Ablations and baselines are run in the following folders:

- Conditional flow matching baseline: `conditional_flow_matching_baseline`. The experiments are run from the `scripts` subfolder.
- Diffusion ablation: `experiments_ablation_diffusion`. This experiment ablates the learnable diffusion term of the SDE and follows the same structure as `experiments`.
- Fully neural model: `experiments_fully_neural`. This experiment uses fully neural models, without additional structure, for the drift and diffusion terms. It follows the same structure as `experiments`.
