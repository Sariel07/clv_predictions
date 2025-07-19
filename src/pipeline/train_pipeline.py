import os
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation  # ← added

class TrainPipeline:
    def __init__(self):
        self.model_save_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'model.pkl')
        )

    def train(self, train_array: np.ndarray, test_array: np.ndarray) -> float:
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            logging.info("Training Linear Regression model")
            model = LinearRegression()
            model.fit(X_train, y_train)

            logging.info("Evaluating model")
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            logging.info(f"✅ Model Performance: MSE = {mse:.4f}, R2 Score = {r2:.4f}")

            save_object(self.model_save_path, model)
            logging.info(f"Model saved at: {self.model_save_path}")

            return r2

        except Exception as e:
            raise CustomException(e, sys)


# 👇 Add this for standalone running
if __name__ == "__main__":
    try:
        logging.info("🚀 Starting Training Pipeline")

        # Initiate transformation
        transformer = DataTransformation()
        train_array, test_array, _ = transformer.initiate_data_transformation(
            train_path="artifacts/train.csv",
            test_path="artifacts/test.csv"
        )

        # Train model
        pipeline = TrainPipeline()
        r2_score_value = pipeline.train(train_array, test_array)
        logging.info(f"✅ Training completed with R2 Score: {r2_score_value:.4f}")

    except Exception as e:
        raise CustomException(e, sys)
