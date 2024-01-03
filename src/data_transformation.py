import sys
from logger import logging
from exception import CustomException
import os
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import save_object
import numpy as np

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")
    logging.info("Defining the preprocessor path")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):  
        try:
            numerical_columns = ["votes", "cost_for_two"]
            categorical_columns = [
                "online",
                "reservations",
                "location",
                "rest_type",
                "type"
            ]
            logging.info("Defining the numeric and categorical features")

            num_pipeline=Pipeline(steps=[
                ("imputing",SimpleImputer(strategy="mean")),
                ("Scaling",StandardScaler())
                ])

            cat_pipeline=Pipeline(steps=[
                ("Imputing",SimpleImputer(strategy="most_frequent")),
                ("Encoding",OneHotEncoder())
                ])
            
            logging.info("Defining the numeric and categorical pipelines")
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ])
            
            logging.info("Defining the preprocessor")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_data=pd.read_csv(train_path)
            train_data=pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            Target = "rating"

            y = train_data[Target]
            X = train_data.drop(Target,axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            preprocessing_obj=self.get_data_transformer_object()
            logging.info("Fetching the preprocessor Successfully.")

            X_train_array=preprocessing_obj.fit_transform(X_train).toarray()
            y_train_array=np.array(y_train)
            X_test_array=preprocessing_obj.transform(X_test).toarray()
            y_test_array=np.array(y_test)
            
            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (X_train_array,y_train_array,X_test_array,y_test_array)

        except Exception as e:
            raise CustomException(e,sys)