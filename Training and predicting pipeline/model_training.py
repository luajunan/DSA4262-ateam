import numpy as np
import pandas as pd
import math
import json

import xgboost as xgb
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from prepare_data import *

def train_model():
    '''
    Trains a new machine learning model using datasets prepared by `prepare_data.py`.
    Processes:
        Recursive feature elimation cross-validation(RFECV) to remove unnecessary features
        GridSearchCV to finetune XGB model for best results
        
    Outputs:
        Trained model's metrics (Accuracy, ROC-AUC, PR-AUC)
        trained_model.json file
    '''
    
    # load data
    directory = "processed_data"
    X_train = pd.read_csv(os.path.join(directory, "trainset.csv"))
    X_test = pd.read_csv(os.path.join(directory, "testset.csv"))
    X_val = pd.read_csv(os.path.join(directory, "valset.csv"))
    
    y_train = X_train.label
    y_test = X_test.label
    y_val = X_val.label

    X_train = X_train.drop(columns=["label"])
    X_test = X_test.drop(columns=["label"])
    X_val = X_val.drop(columns=["label"])
    
    # performing RFE
    print("Performing RFECV to eliminate unimportant features...")
    min_features_to_select = 1

    # To account for weight imbalances
    scale_pos_weight = math.sqrt(y_train.value_counts().values[0]/y_train.value_counts().values[1])

    # create a xgb model --> set random state to ensure reproducibility
    clf_xgb = xgb.XGBClassifier(random_state=4262, use_label_encoder=False, eval_metric="logloss", scale_pos_weight=scale_pos_weight)

    # Recursively eliminate features with cross validation
    rfecv = RFECV(estimator=clf_xgb, cv=5, scoring='roc_auc', n_jobs=-1, verbose=10, step=1, min_features_to_select= min_features_to_select)
    rfecv.fit(X_train, y_train)
    
    X_train = X_train.iloc[:, rfecv.support_] #rfecv.transform(X_train)
    
    # start hyperparameters tuning with gridsearch
    print("Performing GridSearchCV for finetuning hyperparameters...")
    scale_pos_weight = math.sqrt(y_train.value_counts()[0]/y_train.value_counts()[1])

    clf = xgb.XGBClassifier(random_state=4262, colsample_bytree = 0.8, colsample_bynode = 0.8, colsample_bylevel = 0.8, use_label_encoder = False,
                            eval_metric = "logloss", objective = "binary:logistic", scale_pos_weight = scale_pos_weight, n_estimators = 200)

    params = {
            "learning_rate" : [0.03, 0.05, 0.08],
            "max_depth" : [5, 7, 10],
            "min_child_weight" : [1, 3, 5],
            "gamma": [0.5, 1],
            "alpha": [0, 0.5],
            "subsample": [0.8, 1],
            "n_estimators":[300, 500, 700],
            }
            
    best_params = {
    "learning_rate" : [0.03],
    "max_depth" : [7],
    "min_child_weight" : [3],
    "gamma" : [0.5],
    "alpha" : [0],
    "subsample" : [0.8],
    "n_estimators" : [500],
    }
    
    # fine-tuned model:
    # XGBClassifier(alpha=0, base_score=0.5, booster='gbtree', colsample_bylevel=0.8,
              # colsample_bynode=0.8, colsample_bytree=0.8, eval_metric='logloss',
              # gamma=0.5, gpu_id=-1, importance_type='gain',
              # interaction_constraints='', learning_rate=0.03, max_delta_step=0,
              # max_depth=7, min_child_weight=3, missing=nan,
              # monotone_constraints='()', n_estimators=500, n_jobs=8,
              # num_parallel_tree=1, random_state=4262, reg_alpha=0, reg_lambda=1,
              # scale_pos_weight=4.526171050384845, subsample=0.8,
              # tree_method='exact', use_label_encoder=False,
              # validate_parameters=1, verbosity=None)
              
    ## manual finetune, better metrics
    # clf_xgb = xgb.XGBClassifier(alpha=0, base_score=0.5, booster='gbtree', colsample_bylevel=0.8,
              # colsample_bynode=0.8, colsample_bytree=0.8, eval_metric='logloss',
              # gamma=0.5, gpu_id=-1, importance_type='gain',
              # interaction_constraints='', learning_rate=0.05, max_delta_step=0,
              # max_depth=7, min_child_weight=3,
              # monotone_constraints='()', n_estimators=500, n_jobs=8,
              # num_parallel_tree=1, random_state=4262, reg_alpha=0, reg_lambda=1,
              # scale_pos_weight=3, subsample=0.8,
              # tree_method='exact', use_label_encoder=False,
              # validate_parameters=1, verbosity=None)
            
    ## RSCV
    # rscv_model = RandomizedSearchCV(clf, param_distributions=params, n_iter=10, scoring='average_precision', n_jobs=-1, cv=5, verbose=10)

    ## creates the gscv model
    # gscv_model = GridSearchCV(clf, param_grid = params, verbose =10, cv=5, scoring = 'average_precision', n_jobs=-1)
    gscv_model = GridSearchCV(clf, param_grid = best_params, verbose =10, cv=5, scoring = 'average_precision', n_jobs=-1)
    
    ## runs the gscv model
    gscv_model.fit(X_train, y_train)
    
    predictive_features = list(X_train.columns)
    
    # prepares X_val
    X_val = X_val[predictive_features]
    
    ## fit the gscv best model with train data
    print("Fitting finetuned model with train data...")
    clf_xgb = gscv_model.best_estimator_
    clf_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric = "aucpr", verbose=False, early_stopping_rounds=20)
    
    # prepares X_test
    X_test = X_test[predictive_features]
    

    y_pred = clf_xgb.predict(X_test)
    y_pred_proba = clf_xgb.predict_proba(X_test)
    auc_score = roc_auc_score(y_test, y_pred_proba[:,1])
    ap = average_precision_score(y_test, y_pred_proba[:,1])

    print("Performance of trained model on test set:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("AUC-ROC:", auc_score)
    print("PR-ROC:", ap)
    
    # save the model as json
    outdir = './models'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    path = os.path.join(outdir, "trained_model.json")
    
    print(f"Saving model to {path}...")
    clf_xgb.save_model(path)
    
    
    # save rfe features for future predicting
    outdir = './training_features'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    path = os.path.join(outdir, "RFEfeatures.json")
    print(f"Saving RFE features to {path}...")
    with open(path, "w") as fp:
        json.dump(predictive_features, fp)



if __name__ == "__main__":

    # prepare the data first
    print("Processing data...")
    prepare_data(json_input_path=sys.argv[1],
                info_input_path=sys.argv[2])
    
    # then train and output a new model
    print("Training a new model with data...")
    train_model()
    
    print("Done model training.")
    
                
    ## run on terminal:
    ## $ python model_training.py <json_input_path> <info_input_path>
