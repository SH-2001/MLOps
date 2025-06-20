import os
import pandas as pd
import pytest
from unittest.mock import patch, mock_open
from ..pipelines.train import Trainer


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "AnnualPremium": [
                1000,
                1200,
                1100,
                1300,
                1250,
                1400,
                1350,
                1150,
                1450,
                1500,
                1550,
                1600,
                1650,
                1700,
            ],
            "Age": [30, 40, 35, 45, 50, 55, 42, 37, 48, 53, 60, 65, 70, 75],
            "RegionID": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "Gender": [
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
            ],
            "PastAccident": [
                "Yes",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
            ],
            "HasDrivingLicense": [1] * 14,
            "Switch": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def mock_yaml(tmp_path):
    return f"""
    model:
      name: "DecisionTreeClassifier"
      params: {{}}
      store_path: "{tmp_path}"
    """


def test_pipeline_training(df, mock_yaml):
    with patch("builtins.open", mock_open(read_data=mock_yaml)):
        trainer = Trainer()
        X, y = trainer.feature_target_separator(df)
        trainer.train_model(X, y)

        assert trainer.pipeline is not None
        predictions = trainer.pipeline.predict(X)
        assert len(predictions) == len(y)


def test_save_model_creates_directory_and_saves_file(tmp_path, df, mock_yaml):
    with patch("builtins.open", mock_open(read_data=mock_yaml)):
        with patch("joblib.dump") as mock_dump:
            trainer = Trainer()
            X, y = trainer.feature_target_separator(df)
            trainer.train_model(X, y)
            trainer.save_model()

            # Confirm joblib.dump was called with correct path
            model_file = os.path.join(trainer.model_path, "model.pkl")
            mock_dump.assert_called_once_with(trainer.pipeline, model_file)
