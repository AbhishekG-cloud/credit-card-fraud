import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from  src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import StandardScaler

from src.utils import save_object
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer




@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transfromation_config = DataTransformationConfig()
    
    def get_data_transformation_object(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            num_cls = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
       ]
            trans_pipeline = Pipeline(steps=[("Scaler",StandardScaler()),
                                       ("Imputer",SimpleImputer(strategy="mean")),
                                         
                                       ]
                                       )
            
            logging.info("Columns Tsnadard Scaling Completed")
            preprcessor= ColumnTransformer([("tran_pipeline",trans_pipeline,num_cls)])
            return preprcessor
        except Exception as e:
            raise CustomException(e,sys)
        

    def initaite_data_transformation(self,train_path,test_path):
        
        try:
            logging.info("Read train and test data")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("gettiing my preprocessing object")

            preprocessing_obj = self.get_data_transformation_object()
            target_col = "Class"
            input_feature_train_data = train_df.drop(target_col,axis=1)
            output_feature_train_data = train_df[target_col]

            input_feature_test_data = test_df.drop(target_col,axis=1)
            output_feature_test_data = test_df[target_col]
            
            logging.info("Applying preprocessor on train and test dataframes")

            input_feature_train_trnsfmd = preprocessing_obj.fit_transform(input_feature_train_data)
            input_feature_test_trnsfmd = preprocessing_obj.transform(input_feature_test_data)

            train_arr = np.c_[
                input_feature_train_trnsfmd,np.array(output_feature_train_data)
            ]
            test_arr = np.c_[
                input_feature_test_trnsfmd,np.array(output_feature_test_data)
            ]

            logging.info("savde preprocessor object")
            save_object(
                file_path = self.data_transfromation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transfromation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
