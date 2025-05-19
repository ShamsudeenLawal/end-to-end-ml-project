import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

# logging and exception
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("artifacts", "train_data.csv")
    test_data_path: str = os.path.join("artifacts", "test_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_ingestion(self):
        logging.info("Starting Data Ingestion Pipeline")
        try:
            # ingest dataset from sources (local, databases, APIs, etc.)
            logging.info("Reading the dataset as dataframe...")
            data = pd.read_csv("notebooks\data\stud.csv")
            logging.info("Data successfully read.")
            
            # split into train and test data
            logging.info("Splitting the data...")
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42) # as split in the model training notebook
            logging.info("Data successfully splitted.")

            # make artifact directory
            logging.info("Creating datastore...")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            logging.info("Datastore successfully created.")

            # save train and test data into a datastore (artifacts directory) for the next components to use
            logging.info("Saving data into datastore...")
            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data successfully saved into datastore.")

            # log if succesfully
            logging.info("Data Ingestion Completed😁.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                )
        
        except Exception as error:
            raise CustomException(error, sys)

# # usage      
# if __name__=="__main__":
#     ingestion = DataIngestion()
#     train_path, test_path = ingestion.initiate_ingestion()
#     print(train_path, test_path)
