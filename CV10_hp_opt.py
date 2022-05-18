#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:48:40 2022

This python script performs a hyperparamater optimisation on a 
10 fold cross validation study for the prediction of Log(S) via 
various ML models

INPUT: number of trials to be performed
OUTPUT: dataframe containing the best trials in an optuna hyperparameter 
        optimisation study with scoring metrics: R2, RMSE, N1 and N05

@author: gah
"""
import optuna # Package for hyperparameter optimisation
from optuna.samplers import TPESampler # Reproducible search space
import os,re # File path management
import pandas as pd # Store input and output data
import numpy as np # For Y_test
from sklearn import preprocessing # Scaling of training data
from sklearn import ensemble # RF + RF_ET
from scipy.stats import pearsonr # for obtaining R2
from sklearn.model_selection import KFold # CV
import statistics # Results
from sklearn.neural_network import MLPRegressor #ANN
from sklearn.cross_decomposition import PLSRegression #PLS

def main():
    
    '''Function to perform the Optuna optimisation study on each
       of the 4 datasets. Handles file import, setting up and
       execution of the study, and outputs the best results for
       the 4 scoring metrics (R2, RMSE, N01, N05) in a .csv file
       named after the training data. "dataname_hp_opt_metrics.csv"'''
    
    dir = os.getcwd() #get current directory to join to files
    # Import the training data
    AM1 = pd.read_csv(os.path.join(dir,"AM1_full_set_logS.csv"))
    MNDO = pd.read_csv(os.path.join(dir,"MNDO_full_set_logS.csv"))
    PM3 = pd.read_csv(os.path.join(dir,"PM3_full_set_logS.csv"))
    xtb = pd.read_csv(os.path.join(dir,"xtb_full_set_logS.csv"))
    # List of datasets to iterate through
    data_list = [AM1, MNDO, PM3, xtb]
    # Assign output file names
    outfile_names = ['AM1_hp_opt_metrics.csv',
                     'MNDO_hp_opt_metrics.csv',
                     'PM3_hp_opt_metrics.csv',
                     'xtb_hp_opt_metrics.csv']
    # Request number of trials for the study from the user
    n_trials = int(input("Enter number of trials: "))
    
    # Loop through each dataset and perform the Optuna study
    for dataset, outfile in zip(data_list, outfile_names):
        
        # Define an instance of the Objective class initiated with a dataset 
        objective = Objective(dataset)
        # Set random seed so that the results are reproducible
        sampler = TPESampler(seed=10)
        # Create an Optuna study to maximize r2, N1, N05 and minimise RMSE
        study = optuna.create_study(directions=["maximize",
                                                "minimize",
                                                "maximize",
                                                "maximize"],
                                    sampler=sampler)
        # Perform optimisation study with n_trials
        study.optimize(objective, n_trials=n_trials)
        # Get the indexes of the best trials for each study
        best = [study.best_trials[i].number for i in range(len(study.best_trials))]
        # Keep those indexes in the study dataframe
        results = study.trials_dataframe().iloc[best,:]
        # Output the results to a csv file with the corresponding outfile name
        results.to_csv(outfile, index=False)
    
# Define RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
# Define % within certain range
def within_range(list1, list2, range2):
    x=0
    for i in range(len(list2)):
        if (list1[i]-range2)<= list2[i] <= (list1[i]+range2): 
            x+=1
    return((float(x)/(len(list2)))*100)
# Define getting R2 method
def get_R2(R2):
    R2_2=[]
    for i in range(len(R2)):
        x=re.findall('\d\.\d+',str(R2[i]))
        j=float(x[0])
        j=j**2
        R2_2.append(j)
    return(R2_2)

class Objective(object):
    def __init__(self, data):
        self.data = data
        
    def __call__(self, trial):

        #initiate lists to add metrics to (one for )
        RMSEs=[]
        r2s=[]
        N1s=[]
        N05s=[]
        # Suggest a ML model to perform 10 fold CV on
        regressor_name = trial.suggest_categorical("regressor",
                                                   ["ANN", "RF",
                                                    "ET", "PLS",
                                                    "BAG"])
        #import Data and randomise
        X = self.data
        X = X.sample(frac=1).reset_index(drop=True)
        #define k-fold cross validation and make 10 splits
        col_names=X.dtypes.index
        X = np.array(X)
        kf = KFold(n_splits=10)
        #for every split
        for train1, test1 in kf.split(X):
            train=X[train1]
            test=X[test1]
            train=pd.DataFrame(data=train, columns=col_names)
            test=pd.DataFrame(data=test, columns=col_names)
            X_train = train[['MW','volume','G_sol','DeltaG_sol','sol_dip',
                             'Lsolu_Hsolv','Lsolv_Hsolu','SASA','O_charges',
                             'C_charges','Most_neg','Most_pos','Het_charges']]
            y_train = train['LogS']
            X_test = test[['MW','volume','G_sol','DeltaG_sol','sol_dip',
                           'Lsolu_Hsolv','Lsolv_Hsolu','SASA','O_charges',
                           'C_charges','Most_neg','Most_pos','Het_charges']]
            y_test = test['LogS']
            y_test=np.array(y_test)
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Set up values for the hyperparameters:
            # Multi-layer perceptron
            if regressor_name == "ANN":
                # suggest integer for MLP hidden layer size
                MLP_HLS = trial.suggest_int("hidden_layers",
                                            100, 800, log=False)
                # Setup the model=ANN
                model = MLPRegressor(hidden_layer_sizes=MLP_HLS,
                                     max_iter=1000, 
                                     solver = "adam")
                # Train model and get predictions                  
                for f in range(100):
                    try:
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        if np.ptp(preds) == 0:
                            continue
                        break
                    except:
                        continue
            # Random Forest
            elif regressor_name == "RF":
                # suggest integer for RF number of estimators
                RF_trees = trial.suggest_int("n_estimators",
                                             100, 800, log=False)
                # Setup the model=RF
                model = ensemble.RandomForestRegressor(n_estimators=RF_trees,
                                                       n_jobs=-1)
                # Train model and get predictions   
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
            
            # RF ExtraTrees
            elif regressor_name == "ET":
                # suggest integer for RF number of estimators
                ET_trees = trial.suggest_int("n_estimators",
                                             100, 800, log=False)
                # Setup the model=RF_ET
                model = ensemble.ExtraTreesRegressor(n_estimators=ET_trees,
                                                     n_jobs=-1)
                # Train model and get predictions 
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
            # PLS
            elif regressor_name == "PLS":
                # Suggest an integer for number of components
                n_comps = trial.suggest_int("n_components",
                                            7, 13, log=False) #<7 is rubbish
                # Setup the model=PLS2
                model = PLSRegression(n_components=n_comps,
                                     max_iter=1000)
                # Train model and get predictions
                model.fit(X_train, y_train)
                pls2preds = model.predict(X_test)
                
                #convert to float (comes in weird type?)
                preds=[]
                for i in pls2preds:
                    preds.append(float(i))
                    
            # Bagging Regressor
            elif regressor_name == "BAG":
                # Suggest and integer for number of estimators
                BAG_trees = trial.suggest_int("n_estimators",
                                             100, 800, log=False)
                # Setup the model=BaggingRegressor
                model = ensemble.BaggingRegressor(n_estimators=BAG_trees,
                                                  n_jobs=-1)
                # Train model and get predictions
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                
            #evaluate model
            r2s.append(pearsonr(preds, y_test))
            RMSEs.append(rmse(preds, y_test))
            N1s.append(within_range(y_test,preds,1))
            N05s.append(within_range(y_test,preds,0.7))
        #get R2 from Pearson output
        R2s=get_R2(r2s)
        # get average scores
        R2=statistics.mean(R2s)
        RMSE=statistics.mean(RMSEs)
        N1=statistics.mean(N1s)
        N05=statistics.mean(N05s)
        
        return R2, RMSE, N1, N05
        
   
        
if __name__ == "__main__":
    main()