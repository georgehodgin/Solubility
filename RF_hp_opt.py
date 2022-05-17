#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:48:40 2022

This python script performs a hyperparamater optimisation on a 
10 fold cross validation study for the prediction of Log(S) via 
a random forest regressor neural network.

INPUT: - full training set .csv file with LogS column
       - number of trials to be performed
OUTPUT: dataframe containing the best trials in an optuna hyperparameter 
        optimisation study with scoring metrics: R2, RMSE, N1 and N05

@author: gah
"""
import optuna #package for hyperparameter optimisation
from optuna.samplers import TPESampler
import sys,os,re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import ensemble
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import statistics


def main():
    
    
    dir = os.getcwd() #get current directory to join to files
    AM1 = pd.read_csv(os.path.join(dir,"AM1_full_set_logS.csv"))
    MNDO = pd.read_csv(os.path.join(dir,"MNDO_full_set_logS.csv"))
    PM3 = pd.read_csv(os.path.join(dir,"PM3_full_set_logS.csv"))
    xtb = pd.read_csv(os.path.join(dir,"xtb_full_set_logS.csv"))
    data_list = [AM1, MNDO, PM3, xtb]
    outfile_names = ['AM1_RF_opt_metrics.csv',
                     'MNDO_RF_opt_metrics.csv',
                     'PM3_RF_opt_metrics.csv',
                     'xtb_RF_opt_metrics.csv']
    n_trials = int(input("Enter number of trials: "))
    
    for dataset, outfile in zip(data_list, outfile_names):
        
     
        # define an instance of the Objective class initiated with dataset 
        objective = Objective(dataset)
        # set random seed so that the results are reproducible
        sampler = TPESampler(seed=10)
        # create an optuna study to maximize r2, N1, N05 and minimise RMSE
        study = optuna.create_study(directions=["maximize",
                                                "minimize",
                                                "maximize",
                                                "maximize"],
                                    sampler=sampler)
        # perform optimisation study with n_trials
        study.optimize(objective, n_trials=n_trials)
        best = [study.best_trials[i].number for i in range(len(study.best_trials))]
        results = study.trials_dataframe().iloc[best,:]
        results.to_csv(outfile, index=False)
    
#define RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
#define % within certain range
def within_range(list1, list2, range2):
    x=0
    for i in range(len(list2)):
        if (list1[i]-range2)<= list2[i] <= (list1[i]+range2): 
            x+=1
    return((float(x)/(len(list2)))*100)
#define getting R2 method
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
        # suggest integer for MLP hidden layer size
        RF_trees = trial.suggest_int("n_estimators", 100, 1000, log=False)
        #initiate lists to add metrics to (one for )
        RMSEs=[]
        r2s=[]
        N1s=[]
        N05s=[]

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
            #run model
            #Random Forest
            #RF
            model = ensemble.RandomForestRegressor(n_estimators=RF_trees,
                                                   n_jobs=-1)
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