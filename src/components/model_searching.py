# Basic import
import os
import sys
import joblib
import warnings
import pandas as pd


# exception and logging
from src.logger import logging
from src.exception import CustomException

# modeling
from sklearn.linear_model import LinearRegression

# configurations
from dataclasses import dataclass
from src.models_configs import estimators_configurations

# hyperparameter tuning functions
from sklearn.model_selection import RandomizedSearchCV, \
                                    cross_val_score, \
                                    cross_val_predict

@dataclass
class ModelSearchingConfig:
    model_path: str = os.path.join("artifacts", "regressor.joblib")
    metrics_path: str = os.path.join("artifacts", "metrics.csv")
    search_result_path: str = os.path.join("artifacts", "search_results.joblib")


class ModelSearching:
    def __init__(self):
        self.model_config = ModelSearchingConfig()

    def initiate_model_search(self, train_data, test_data, cv=5, n_iter=30):
        
        logging.info("Starting model searching...")
        
        try:
            # Separate into features and labels
            Xtrain = train_data.values[:, :-1]
            ytrain = train_data.values[:, -1].ravel()

            Xtest = test_data.values[:, :-1]
            ytest = test_data.values[:, -1].ravel()

            model_names = []
            validation_scores = []
            test_scores = []

            # tuning model
            warnings.filterwarnings("ignore")
            for i, (estimator_name, estimator_config ) in enumerate(estimators_configurations.items()):
                random_search = RandomizedSearchCV(
                    estimator=estimator_config["estimator"],
                    param_distributions=estimator_config["params"],
                    n_iter=n_iter,  # number of combinations to try for each model
                    cv=cv,
                    refit=True,
                    # scoring="neg_root_mean_squared_error",
                    random_state=42
                    )
                
                random_search.fit(Xtrain, ytrain)
                
                val_score = random_search.best_score_

                test_score = random_search.score(Xtest, ytest)

                # append necessary details
                model_names.append(estimator_name)
                validation_scores.append(val_score)
                test_scores.append(test_score)

            logging.info("Collecting and saving metrics")
            # collect results as dataframe
            metrics = pd.DataFrame(data={"validation_scores": validation_scores, "test_scores": test_scores},
                                    index=model_names)
            
            best_score = metrics["validation_scores"].max()
            
            # saving metrics into datastore
            metrics.to_csv(self.model_config.metrics_path, index=True, header=True)
            logging.info("Metrics successfully saved.")

            # writing model into datastore(artifacts)
            logging.info("Saving model to datastore...")
            joblib.dump(random_search.best_estimator_, self.model_config.model_path)
            logging.info("Model successfully saved.")

            # write search results into datastore
            logging.info("Saving search results...")
            joblib.dump(random_search.cv_results_, filename=self.model_config.search_result_path) # saving the entire random search
            # joblib.dump(random_search.cv_results_, filename=self.model_config.search_result_path) # saving only the result
            logging.info("Search result saved.")

            logging.info("Model searching completed.")

            return best_score

        except Exception as err:
            raise CustomException(err, sys)
        
# # testing
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
#     search = ModelSearching()
#     best_score = search.initiate_model_search(transformed_train_data, transformed_test_data, n_iter=5)
#     print(best_score)
