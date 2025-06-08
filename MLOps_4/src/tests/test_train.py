import os
import pandas as pd
from unittest.mock import patch, mock_open

import pytest

from ..pipelines.train import Trainer


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "AnnualPremium": [1000, 1200, 1100, 1300, 1250, 1400, 1380, 900],
            "Age": [30, 40, 35, 45, 50, 55, 56, 31],
            "RegionID": [1, 2, 1, 2, 1, 2, 2, 1],
            "Gender": [
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Male",
            ],
            "PastAccident": [
                "Yes",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
                "No",
                "Yes",
            ],
            "HasDrivingLicense": [1, 1, 1, 1, 1, 1, 1, 1],
            "Switch": [0, 1, 0, 1, 0, 1, 1, 0],
        }
    )


@pytest.fixture
def mock_yaml(tmp_path):
    return f"""
    model:
      name: "RandomForestClassifier"
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
