# Codebase for Oh SnapMMD! Forecasting stochastic dynamics beyond the Schrödinger bridge's end
Renato Berlinghieri*, Yunyi Shen*, Jialong Jiang and Tamara Broderick
*: equal contribution
## Structure
There are three main folders in this repo, `experiments`, `data` and `package`. In the `package` we implmented our method as a python package `snapMMD` to be installed. After installation one can run experiments in `experiments` that depends on files in `data`.

## Environment
To run our code we suggest the following procedure. 

1) Create a virtual environment with Python version Python 3.8.16, e.g., in conda `conda create -n "snapmmd" python=3.8.16 ipython` and activate `conda activate snapmmd`. This should work with other versions of python with torch and torchsde installed but we did not test.  
2) Install all dependencies needed from `requirements.txt`, by running `pip install -r requirements.txt`
3) Install our package snapMMD by running `pip install ./package` 

## Experiments
To reproduce our experiment, run scripts in `experiments`, each scripts take one argument on which experiment to run. 

There are four subfolders in `experiments`, 
- `classic`: Classic SDEs. There are two experiments: `LV` for Lotka-Volterra, `Repressilator` for repressilator. The main script is `classic_sde.py`, e.g., `python classic_sde.py LV` will run the LV experiment. 
- `missingobs`: Missing observations. There is one experiment: `Repressilator` for repressilator with missing protein observations. The main script is `missing_data.py`. 
- `mlp`: Semiparametric model. There is one experiment: `Repressilator` for repressilator with an MLP "activation" function. The main script is `MLP.py`.
- `realdata`: Real data experiments. There are two experiments: `GoM` for vortex in Gulf of Mexico and `pbmc` for single cell dataset. The main script is `realdata.py`


