
import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from  src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model
from src.components.hyperparamter_tuning import Tunner


from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from xgboost import XGBClassifier



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_training(self,train_array,test_array):
        try:
            logging.info("Splitting training and test array")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )
            logging.info("Getting best Estimator and Params from Tunning")
            ## do tunning  only once
            ##tunning = Tunner()
            ##best_results=tunning.tunner(X_train = X_train,y_train=y_train)
            ##best_model = best_results["best_estimator"]
            
            ## best model is Xgboost              
            ##'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05 for Xg boost
            model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate= 0.05)
            model.fit(X_train,y_train)


            logging.info("Saving model pickle")
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= model
                        )
            model.fit(X_train,y_train)
            y_proba = model.predict_proba(X_test)[:,1]
            thresh = 0.169
            y_pred = (y_proba>=thresh).astype(int)
            

            report = classification_report(y_test,y_pred)
            return report



            
        except Exception as e:
            raise CustomException(e,sys)
        

