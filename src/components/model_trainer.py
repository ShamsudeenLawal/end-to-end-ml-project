# Basic import
import os
import sys
import joblib
import pandas as pd

# exception and logging
from src.logger import logging
from src.exception import CustomException

# modeling
from sklearn.linear_model import LinearRegression

# evaluation
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# configurations
from dataclasses import dataclass

@dataclass
class ModelTrainingConfig:
    model_path: str = os.path.join("artifacts", "regressor.joblib")
    metrics_path: str = os.path.join("artifacts", "metrics.csv")


class ModelTraining:
    def __init__(self):
        self.model_config = ModelTrainingConfig()

    def train_model(self, X, y):
        self.model = LinearRegression().fit(X, y)
        return self

    def evaluate_model(self, X, y):
        ypred = self.model.predict(X)
        r2 = r2_score(y, ypred)
        rmse = root_mean_squared_error(y, ypred)
        mae = mean_absolute_error(y, ypred)

        metrics = {"R2": r2,
                   "RMSE": rmse,
                   "MAE": mae}

        metrics = pd.DataFrame(data=metrics, index=["Linear Regression"])

        return metrics

    def train_and_evaluate_model(self, train_data, test_data):
        # start training and evaluation
        logging.info("Starting model training and evaluation...")
        try:
            # Separate into features and labels
            Xtrain = train_data.values[:, :-1]
            ytrain = train_data.values[:, -1].ravel()

            Xtest = test_data.values[:, :-1]
            ytest = test_data.values[:, -1].ravel()

            # train model
            logging.info("Training model...")
            self.train_model(Xtrain, ytrain)
            logging.info("Model successfully trained.")

            # evaluate model
            logging.info("Evaluating model...")
            test_metrics = self.evaluate_model(Xtest, ytest)
            logging.info("Model evaluation completed.")

            # writing model into datastore(artifacts)
            logging.info("Saving model to datastore...")
            joblib.dump(self.model, self.model_config.model_path)
            logging.info("Model successfully saved.")

            # saving metrics into datastore
            logging.info("Saving metrics to datastore...")
            test_metrics.to_csv(self.model_config.metrics_path, index=False, header=True)
            logging.info("Metrics successfully saved.")

            logging.info("Model training and evaluation completed.")

            return test_metrics

        except Exception as err:
            raise CustomException(err, sys)

# # usage
# if __name__ == "__main__":
#     from src.components.data_ingestion import DataIngestion
#     from src.components.data_transformation import DataTransformation

#     # ingesting data
#     data_ingestion = DataIngestion()
#     train_data_path, test_data_path = data_ingestion.initiate_ingestion()
#     # transforming data
#     data_transformer = DataTransformation()
#     transformed_train_data, transformed_test_data = data_transformer.initiate_transformation(train_data_path, test_data_path)
#     # model training and evaluation
#     trainer = ModelTraining()
#     metrics = trainer.train_and_evaluate_model(transformed_train_data, transformed_test_data)
#     print(metrics)
