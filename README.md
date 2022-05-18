# Solubility
hyperparameter optimisation of solubility prediction via various ML methods

## Description
The Python script CV10_hp_opt.py performs a hyperparameter optimisation 
study for each dataset contained within the CV_data directory. The Optuna
package in Python is used to perform the optimisation on the 4 model
performance metrics (R^2, RMSE, % within 1%, % within 0.7%) for a 10 fold
cross validation. The output files from CV10_hp_opt.py are stored in the
Results directory.

## Usage
Clone the repository: 
```{bash}
git clone git@github.com:georgehodgin/Solubility.git
```

Build the conda environment:
```{bash}
cd Solubility
conda env create -f conda_environment.yml
```
Activate the conda environment:
```{bash}
conda activate hpopt
```
Run the code:
```{bash}
python CV10_hp_opt.py
```
