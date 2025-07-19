import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Returns the ColumnTransformer preprocessing object
        """
        try:
            numerical_columns = ["Quantity", "UnitPrice"]
            categorical_columns = ["Country"]

            logging.info("Creating transformation pipelines")

            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            # Combined column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Preprocessor object created")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading training and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Clean and validate both datasets
            logging.info("Cleaning train and test datasets")
            cleaned_dfs = []
            for df in [train_df, test_df]:
                df.dropna(subset=["Quantity", "UnitPrice", "Country", "CLV"], inplace=True)
                df = df[df["Quantity"] > 0]
                cleaned_dfs.append(df)

            train_df, test_df = cleaned_dfs

            # Initialize preprocessor
            preprocessing_obj = self.get_data_transformer_object()
            target_column = "CLV"

            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            # Fit and transform
            logging.info("Fitting and transforming data")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info(f"Transformed train shape: {input_feature_train_arr.shape}")
            logging.info(f"Transformed test shape: {input_feature_test_arr.shape}")

            # Concatenate target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor saved to artifacts/preprocessor.pkl")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

# Optional: test run directly
if __name__ == "__main__":
    transformer = DataTransformation()
    transformer.initiate_data_transformation("artifacts/train.csv", "artifacts/test.csv")
