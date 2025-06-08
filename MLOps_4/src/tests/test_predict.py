import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from sklearn.dummy import DummyClassifier
from ..pipelines.predict import Predictor


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "feature1": [0.1, 0.2, 0.3, 0.4],
            "feature2": [1, 0, 1, 0],
            "target": [0, 1, 0, 1],
        }
    )


mock_yaml = """
    model:
      name: "RandomForestClassifier"
      params: {}
      store_path: "mock_model_path"
    """


@patch("joblib.load")
@patch("builtins.open", new_callable=mock_open, read_data=mock_yaml)
def test_feature_target_separator(mock_file, mock_joblib_load, sample_data):
    mock_model = MagicMock()
    mock_model.predict.return_value = [0, 1, 0, 1]
    mock_joblib_load.return_value = mock_model
    pred = Predictor()
    X, y = pred.feature_target_separator(sample_data)
    assert list(X.columns) == ["feature1", "feature2"]
    assert list(y) == [0, 1, 0, 1]


def test_evaluate_model(sample_data):
    # Split sample data
    X = sample_data[["feature1", "feature2"]]
    y = sample_data["target"]

    # Dummy model simulating model.pkl
    dummy_model = DummyClassifier(strategy="most_frequent")
    dummy_model.fit(X, y)

    # Mock YAML config and joblib.load
    mock_yaml = """
    model:
      store_path: "fake_path"
    """

    with patch("builtins.open", mock_open(read_data=mock_yaml)):
        with patch("joblib.load", return_value=dummy_model):
            pred = Predictor()
            accuracy, class_report, roc_auc = pred.evaluate_model(X, y)

            assert 0.0 <= accuracy <= 1.0
            assert isinstance(class_report, str)
            assert 0.0 <= roc_auc <= 1.0
