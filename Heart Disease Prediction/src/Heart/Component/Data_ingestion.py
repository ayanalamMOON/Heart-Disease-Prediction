import os
import sys
import numpy as np
import pandas as pd
from Heart.logger import logging
from Heart.exception import customeception
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("Artifacts","heart.csv")
    train_data_path:str = os.path.join("Artifacts","train_data.csv")
    test_data_path:str = os.path.join("Artifacts", "test_data.csv")


class DataIngestion:
    def __init__(self)
        self.ingestion_config = DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            data = pd.read_csv("Notebook_Experiments\Data\heart.csv")
            logging.info("Read Data from the CSV file")


            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Created Raw Data File")

            logging.info("Splitting Data into Train and Test Data")
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            logging.info("Data Splitting is done")

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index = False)
            logging.info("Created Train and Test Data files")
            logging.info("Data Ingestion Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Exception Occurred in Data Ingestion")
            raise customexception(e,sys)