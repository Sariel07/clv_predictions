import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process for CLV project.")

        try:
            # Load raw Excel data
            df = pd.read_excel("notebook/data/Online Retail.xlsx")

            logging.info("Excel data loaded successfully.")

            # Drop rows with missing CustomerID or UnitPrice or Quantity
            df.dropna(subset=["CustomerID", "Quantity", "UnitPrice"], inplace=True)

            # Remove returns (negative quantity)
            df = df[df["Quantity"] > 0]

            # Create CLV column
            df["CLV"] = df["Quantity"] * df["UnitPrice"]

            # Create artifacts directory if not exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw cleaned data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw cleaned data saved.")

            # Train-test split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            train_df.to_csv(self.ingestion_config.train_data_path, index=False)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed. Train and test sets saved.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Error occurred during data ingestion.")
            raise CustomException(e, sys)

# Optional: To run standalone for testing
if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print(f"Train saved at: {train_path}\nTest saved at: {test_path}")
