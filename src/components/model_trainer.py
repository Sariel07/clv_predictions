import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'model.pkl')
    )

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def train_model(self, train_array: np.ndarray, test_array: np.ndarray):
        try:
            logging.info("Splitting input and target features from training and test arrays")

            # Last column is target (CLV)
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            logging.info("Training Linear Regression model")
            model = LinearRegression()
            model.fit(X_train, y_train)

            logging.info("Evaluating model performance")
            predictions = model.predict(X_test)

            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)

            logging.info(f"✅ R2 Score: {r2:.4f}")
            logging.info(f"✅ MAE: {mae:.4f}")
            logging.info(f"✅ MSE: {mse:.4f}")

            # Save the trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logging.info(f"Model saved to {self.model_trainer_config.trained_model_file_path}")
            return r2  # or return full metrics if you want

        except Exception as e:
            raise CustomException(e, sys)

# Optional test run
if __name__ == "__main__":
    # This assumes you've already called data_transformation.py
    from src.components.data_transformation import DataTransformation

    transformer = DataTransformation()
    train_arr, test_arr, _ = transformer.initiate_data_transformation(
        "artifacts/train.csv", "artifacts/test.csv"
    )

    trainer = ModelTrainer()
    trainer.train_model(train_arr, test_arr)
