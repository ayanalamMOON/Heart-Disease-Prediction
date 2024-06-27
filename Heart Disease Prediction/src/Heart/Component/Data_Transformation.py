import os
from platform import system
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from Heart.logger import logging # type: ignore
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.Heart.utils.utils import save_object
from sklearn.compose import ColumnTransformer
from Heart.exception import customExeception
from sklearn.preprocessing import StandardScaler


@dataclass

class Data_TransformationConfig:
    preprocessor_obj_file_path=os.path.join('Artifacts', 'Preprocessor.pkl')


class Data_Transformation:
    def __init__(self):
        self.data_transformation_config = Data_TransformationConfig()


    def get_data_transformation(self):

        try:
            logging.info('Data Transformation Initiated')
            
            numerical_cols =  ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            logging.info('Numerical Pipline Initiated')


            ## Numerical Pipeline

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

        preprocessor = ColumnTransformer([('num_pipeline',num_pipeline,numerical_cols)])
        return preprocessor
    
except Exception as e:
    logging.info("Exceptption occured in the initiate_datatransformation")
    raise customexeception(e,sys)


def initialize_data_transformation(self, train_path, test_path):
    try:
        train_df=pd.read_csv(train_path)
        test_df=pd.read_csv(test_path)

        logging.info("read train and test data complete")
        logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
        logging.info(f'Test Dataframe Head : \n{test_df.head().to_sttring()}')

        preprocessing_obj = self.get_data_transformation()

        target_column_name = 'target'
        drop_columns = [target_column_name]

        input_features_df = train_df.drop(columns=drop_columns,axis = 1)
        target_feature_train_df = train_df[target_column_name]
        input_features_test_df = test_df.drop(columns=drop_columns, axis= 1)
        target_feature_test_df = test_df[target_column_name]
        logging.info("Splitting the data into input fearures and target features complete")

        input_features_train_arr = preprocessing_obj.fit_transform(input_features_df)
        input_features_test_arr = preprocessing_obj.transform(input_features_test_df)
        logging.info("Applying Preprocessing object to tain and test data complete")

        train_arr = np.c_[input_features_train_arr,np.array(target_feature_train_df)]
        test_arr = np.c_[input_features_testarr,np.array(target_feature_test_df)]

        save_objest(
            file_path=self.data_transformation_config.preprocessor_obj_filepath,
            obj=preprocessing_obj
        )

        logging.info("Preprocessing pickle file saved")
        return(tari_test_arr, test_arr)
    
    except Exception as e: 
        logging.info("Exceptption occured in the initiate_datatransformation")
        raise customExeception(e,system)

