import os
import sys
import pandas as pd
import numpy as np

from  src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model


from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.model_selection  import RandomizedSearchCV,GridSearchCV
from xgboost import XGBClassifier


class Tunner:
    def tunner(self,X_train,y_train):
       

        models = {
    
        "LogisticRegressionCV": LogisticRegressionCV(),
        "AdaBoostClassifier": AdaBoostClassifier(),
        "XGBClassifier": XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False
        )
        }

        param_grids = {

        "LogisticRegressionCV": {
        "Cs": [3, 5, 10, 20],
        "cv": [3, 5],
        "penalty": ["l2"],
        "solver": ["lbfgs"]
        },

        "AdaBoostClassifier": {
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.001, 0.01, 0.1, 0.5, 1.0]
        },

        "XGBClassifier": {
        "max_depth" :[3, 5, 7],

        "learning_rate" : [0.05, 0.1],

        "n_estimators" :[100, 200]
        }}
        best_models = {}

        for name, model in models.items():

            print(f"\nTuning {name}...")

            if name == "XGBClassifier":
                search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grids[name],
            n_iter=50,
            scoring='roc_auc',
            cv=5,
            n_jobs=-1,
            verbose=2,
            random_state=42
        )
            else:
                search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            scoring='roc_auc',
            cv=5,
            n_jobs=-1,
            verbose=2
        )

            search.fit(X_train, y_train)

        best_models[name] = search.best_estimator_

        print(f"Best Params: {search.best_params_}")
        print(f"Best CV Score: {search.best_score_:.4f}")
## best model in Xgboost              
##'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05 for Xg boost