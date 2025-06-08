import pytest
import pandas as pd
from unittest.mock import patch
from sklearn.dummy import DummyClassifier
from ..pipelines.predict import Predictor


@pytest.fixture
def sample_data():
    # Create a small dummy dataframe similar to your expected data format
    data = {
        "feature1": [0.1, 0.2, 0.3, 0.4],
        "feature2": [1, 0, 1, 0],
        "target": [0, 1, 0, 1],
    }
    return pd.DataFrame(data)


def test_feature_target_separator(sample_data):
    pred = Predictor()
    X, y = pred.feature_target_separator(sample_data)
    assert list(X.columns) == ["feature1", "feature2"]
    assert list(y) == [0, 1, 0, 1]


def test_evaluate_model(sample_data):
    with patch("pipelines.predict.Predictor.load_model") as mocked_load:
        mocked_load.return_value = DummyClassifier(
            strategy="most_frequent"
        ).fit(sample_data.iloc[:, :-1], sample_data.iloc[:, -1])
        pred = Predictor()  # loads dummy model now

        X, y = pred.feature_target_separator(sample_data)
        accuracy, class_report, roc_auc = pred.evaluate_model(X, y)

        assert 0.0 <= accuracy <= 1.0
        assert 0.0 <= roc_auc <= 1.0
