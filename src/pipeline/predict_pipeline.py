import os
import sys
import numpy as np
import pandas as pd
import joblib

from src.exception import CustomException
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        # Absolute paths for artifacts
        self.preprocessor_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'preprocessor.pkl')
        )
        self.model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'model.pkl')
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        try:
            logging.info("🔄 Loading preprocessor and model for prediction...")
            preprocessor = joblib.load(self.preprocessor_path)
            model = joblib.load(self.model_path)

            logging.info("✅ Preprocessing input features")
            transformed_features = preprocessor.transform(features)

            logging.info("🔍 Making predictions")
            predictions = model.predict(transformed_features)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)
