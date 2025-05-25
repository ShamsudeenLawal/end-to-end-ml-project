
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_searching import ModelSearching

class TrainingPipeline:
    def __init__(self):
        self.ingestion_pipeline = DataIngestion()
        self.transformation_pipeline = DataTransformation()
        self.search_pipeline = ModelSearching()

    def initiate_model_training(self, n_iter=30):
        # ingesting data
        train_data_path, test_data_path = self.ingestion_pipeline.initiate_ingestion()

        # transforming data
        transformed_train_data, transformed_test_data = self.transformation_pipeline.initiate_transformation(train_data_path, test_data_path)

        # model training and evaluation
        best_score = self.search_pipeline.initiate_model_search(transformed_train_data, transformed_test_data, n_iter=n_iter)
        print(best_score)

if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    training_pipeline.initiate_model_training(n_iter=30)

