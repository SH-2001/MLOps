import os
import pandas as pd
from unittest.mock import patch, mock_open
from ..pipelines.train import Trainer


def test_pipeline_training():
    # Minimal sample input DataFrame
    df = pd.DataFrame(
        {
            "AnnualPremium": [1000, 1200, 1100, 1300, 1250, 1400],
            "Age": [30, 40, 35, 45, 50, 55],
            "RegionID": [1, 2, 1, 2, 1, 2],
            "Gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
            "PastAccident": ["Yes", "No", "Yes", "No", "Yes", "No"],
            "HasDrivingLicense": [1, 1, 1, 1, 1, 1],
            "Switch": [0, 1, 0, 1, 0, 1],
        }
    )

    mock_yaml = """
    model:
      name: "RandomForestClassifier"
      params: {}
      store_path: "mock_model_dir"
    """

    with patch("builtins.open", mock_open(read_data=mock_yaml)):
        trainer = Trainer()
        X, y = trainer.feature_target_separator(df)
        trainer.train_model(X, y)

        assert trainer.pipeline is not None
        predictions = trainer.pipeline.predict(X)
        assert len(predictions) == len(y)


def test_save_model_creates_directory_and_saves_file(tmp_path):
    mock_yaml = f"""
    model:
        name: "RandomForestClassifier"
        params: {{}}
        store_path: "{tmp_path}"
    """

    df = pd.DataFrame(
        {
            "AnnualPremium": [1000, 1200],
            "Age": [30, 40],
            "RegionID": [1, 2],
            "Gender": ["Male", "Female"],
            "PastAccident": ["Yes", "No"],
            "HasDrivingLicense": [1, 1],
            "Switch": [0, 1],
        }
    )

    with patch("builtins.open", mock_open(read_data=mock_yaml)):
        with patch("joblib.dump") as mock_dump:
            trainer = Trainer()
            X, y = trainer.feature_target_separator(df)
            trainer.train_model(X, y)
            trainer.save_model()

            # Confirm joblib.dump was called with correct path
            model_file = os.path.join(trainer.model_path, "model.pkl")
            mock_dump.assert_called_once_with(trainer.pipeline, model_file)
