# Codebase for Oh SnapMMD! Forecasting stochastic dynamics beyond the Schrödinger bridge's end
Renato Berlinghieri*, Yunyi Shen*, Jialong Jiang and Tamara Broderick
*: equal contribution
## Structure
There are three main folders in this repo, `experiments`, `data` and `package`. In `package`, we implement our method as a Python package called `snapMMD` to be installed. After installation, one can reproduce our experiments by running scripts in the `experiments` folder that depends on files in the `data` folder.

## Environment
To run our code we suggest the following procedure:

1) Create a virtual environment with Python version Python 3.8.16, e.g., in conda `conda create -n "snapmmd" python=3.8.16 ipython` and activate it by running `conda activate snapmmd`
2) Install all dependencies needed from `requirements.txt`, by running `pip install -r requirements.txt`
3) Install our package snapMMD by running `pip install ./package` from the base folder

## Experiments
To reproduce our experiments, you can find scripts in `experiments`. Each script takes one argument about which experiment to run. 

There are four subfolders in `experiments`:
- `classic`: experiments for which the data generating processes are classic SDEs systems. In this folder there are two experiments: `LV` for the Lotka-Volterra system, and `Repressilator` for the repressilator system. We refer to our manuscript for discussion about these two dynamical systems (Appendix D.5 and D.6). The main script to run these is `classic_sde.py`. E.g., if you are interested in reproducing our experiments for the Lotka-Volterra (`LV`) system, you can run `python classic_sde.py LV`. 
- `missingobs`: experiments for which we have data with incomplete state measurements. For this scenario we have one experiment, `Repressilator`, which is about the repressilator system with missing protein observations. The results for this experiment can be reproduced by running the script `missing_data.py`. More details about this experiment can be found in Appendix D.8 in our paper. 
- `mlp`: experiments for which we use a semiparametric model instead of the data generating process as our model family. See Section 4 in the paper for further details on this difference, as well as Appendix D.7 for more details on this experiment setup.  Also in this folder we have only one experiment, `Repressilator`, for the repressilator with an MLP "activation" function. The main script is `MLP.py`.
- `realdata`: experiments for which we do not have access to the data generating process as the data are real data observations. Here we have two experiments: `GoM` for vortex in Gulf of Mexico (Appendix D.9) and `pbmc` for a single cell dataset (Appendix D.10). The main script is `realdata.py`, and as for the `classic` experiment you will still select the experiment to run by passing the name as an argument. E.g., to run the `pbmc` experiment you can run `python realdata.py pbmc`. 


