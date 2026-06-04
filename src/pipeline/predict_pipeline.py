import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.utils import load_object
@dataclass
class CustomData:
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

    def get_data_as_dataframe(self):
        return pd.DataFrame([self.__dict__])

class PredPipeline():
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            scaled_data = preprocessor.transform(features)
            preds = model.predict(scaled_data)


            return preds
        

        except Exception as e:
            raise CustomException(e,sys)
        

        


