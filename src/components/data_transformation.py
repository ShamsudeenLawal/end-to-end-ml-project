# Basic importation
import os
import sys
import numpy as np
import pandas as pd

# preprocessing modules
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# config creation modules
from dataclasses import dataclass

# pickling into file
import joblib
# import dill

# logging and exception
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.joblib")

class DataTransformation:
    def __init__(self):
        self.tranformation_config = DataTransformationConfig()

    def instantiate_preprocessor(self):
        # define features and labels
        self.label = ["math_score"]
        self.numerical_features = ["reading_score", "writing_score"]
        self.categorical_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
        
        # create preprocessor pipelines
        numerical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                             ("scaler", StandardScaler())])
        
        categorical_pipeline = Pipeline(steps=[("cat_imputer", SimpleImputer(strategy="most_frequent")),
                                             ("encoder", OneHotEncoder())])
        
        # combine pipelines into transformer
        preprocessor = ColumnTransformer(transformers=[
            ("numerical_pipeline", numerical_pipeline, self.numerical_features),
            ("categorical_pipeline", categorical_pipeline, self.categorical_features)
        ])

        return preprocessor
    
    def initiate_transformation(self, train_path, test_path):
        
        # starting transformation
        logging.info("Starting data transformation...")
        try:
            # instantiate preprocessor
            logging.info("Instantiating preprocessor")
            preprocessor = self.instantiate_preprocessor()
            logging.info("Preprocessor instantiated.")
            
            # read data
            logging.info("Reading train and test data from datastore.")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info("Data read successfully.")
            
            # separate features and labels
            ytrain = train_data[self.label]
            Xtrain = train_data.drop(columns=self.label)

            ytest = test_data[self.label]
            Xtest = test_data.drop(columns=self.label)

            # fit and transform features
            logging.info("Fitting preprocessor and transforming data...")
            preprocessor = preprocessor.fit(Xtrain)
            transformed_Xtrain = preprocessor.transform(Xtrain)
            transformed_Xtest = preprocessor.transform(Xtest)

            # combine transformed features and label
            transformed_train_data = np.c_[transformed_Xtrain, ytrain]
            transformed_test_data = np.c_[transformed_Xtest, ytest]
            logging.info("Fitting and transformation completed successfully.")

            # save fitted preprocessor into datastore(artifacts)
            logging.info("Saving preprocessor into datastore...")
            joblib.dump(preprocessor, filename=self.tranformation_config.preprocessor_path)
            logging.info("Preprocessor successfully saved.")

            # data transformation completed
            logging.info("Data transformation completed.")

            # # returning numpy arrays
            # return (transformed_train_data, transformed_test_data)
        
            # returning dataframe (this is okay for scikit-learn models, but for tensorflow where data can be 3D, one should return numpy)
            transformed_train_data = pd.DataFrame(transformed_train_data, columns=preprocessor.get_feature_names_out().tolist() + self.label)
            transformed_test_data = pd.DataFrame(transformed_test_data, columns=preprocessor.get_feature_names_out().tolist() + self.label)
            return (transformed_train_data, transformed_test_data)

        except Exception as err:
            raise CustomException(err, sys)
        
# # usage
# if __name__ == "__main__":
#     from src.components.data_ingestion import DataIngestion
#     # ingesting data
#     data_ingestion = DataIngestion()
#     train_data_path, test_data_path = data_ingestion.initiate_ingestion()
#     # transforming data
#     data_transformer = DataTransformation()
#     transformed_train_data, transformed_test_data = data_transformer.initiate_transformation(train_data_path, test_data_path)
#     print(type(transformed_train_data))
#     print(transformed_train_data.head(5))
