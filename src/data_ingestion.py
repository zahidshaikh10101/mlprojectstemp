from logger import logging
from exception import CustomException
import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
import mysql.connector as connection
from data_transformation import DataTransformation,DataTransformationConfig
from model_trainer import ModelTrainerConfig,ModelTrainer


@dataclass
class DataIngestionConfig:
    raw_data_path=os.path.join(os.getcwd(),"artifacts","data.csv")
    train_data_path=os.path.join(os.getcwd(),"artifacts","train.csv")
    test_data_path=os.path.join(os.getcwd(),"artifacts","test.csv")
    logging.info("give the row data,train and test path to store the data.")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")
        try:
            #df = pd.read_csv(R"notebook\data\Updated_Zomato_Data.csv")
            mydb = connection.connect(host="localhost", database = 'zomato',user="root", passwd="Zarss@786",use_pure=True)
            logging.info("set the connection with the mysql for read the data")
            query = "select * from zomato.zomato_data;"
            logging.info(f"select the column which we need for the dataset: {query}")
            result_dataFrame = pd.read_sql(query,mydb)
            mydb.close()
            df = result_dataFrame
            logging.info("Read as Dataframes")
            train_data,test_data = train_test_split(df,test_size=0.2,random_state=42)
            logging.info("Train Test split initiated")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            logging.info("Created the Artifacts folder for the save the train and test data")

            df.to_csv(self.ingestion_config.raw_data_path)
            train_data.to_csv(self.ingestion_config.train_data_path)
            test_data.to_csv(self.ingestion_config.test_data_path)
            
            logging.info("Data Ingestion is completed")
            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    data_ingestion_obj=DataIngestion()
    train_path,test_path=data_ingestion_obj.initiate_data_ingestion()
    data_transformation_obj=DataTransformation()
    X_train_array,y_train_array,X_test_array,y_test_array=data_transformation_obj.initiate_data_transformation(train_path,test_path)
    model_trainer_obj=ModelTrainer()
    model_trainer_final=model_trainer_obj.initiate_model_trainer(X_train_array,y_train_array,X_test_array,y_test_array)
    print(model_trainer_final)