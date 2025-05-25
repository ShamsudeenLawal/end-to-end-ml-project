
import os
import joblib
import pandas as pd

class CustomData:
    def __init__(self, gender,
                 race_ethnicity,
                 parental_level_of_education,
                 lunch,
                 test_preparation_course,
                 reading_score,
                 writing_score):
        
        self.gender = gender
        self.lunch = lunch
        self.reading_score = reading_score
        self.writing_score = writing_score
        self.race_ethnicity = race_ethnicity
        self.test_preparation_course = test_preparation_course
        self.parental_level_of_education = parental_level_of_education

    def get_data_as_df(self):
        data = {
            "gender": [self.gender],
            "race_ethnicity": [self.race_ethnicity],
            "parental_level_of_education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test_preparation_course": [self.test_preparation_course],
            "reading_score": [self.reading_score],
            "writing_score": [self.writing_score]
        }

        df_data = pd.DataFrame(data=data)

        return df_data
    

class PredictPipeline:
    def __init__(self, ):
        self.model = joblib.load(os.path.join("artifacts", "regressor.joblib"))
        self.preprocessor = joblib.load(os.path.join("artifacts", "preprocessor.joblib"))


    def predict(self, features):
        transformed_data = self.preprocessor.transform(features)
        prediction = self.model.predict(transformed_data).ravel()

        return int(prediction[0])
    


        