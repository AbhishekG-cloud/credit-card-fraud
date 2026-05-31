
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
from xgboost import XGBClassifier

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __int__(self):
        self.model_triner_config = ModelTrainerConfig()


    def initiate_training(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Splitting training and test array")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )
            models = {
            
            "LogisticRegressionCV":LogisticRegressionCV(),
            "XGBClassifier":XGBClassifier(),
            "AdaBoostClassifier":AdaBoostClassifier()


            }
            model_report:dict = evaluate_model(X_train= X_train,y_train = y_train,
                                                X_test = X_test,y_test = y_test,model = models)
            
        except Exception as e:
            raise CustomException(e,sys)
        

