import pandas as pd
from unittest.mock import mock_open, patch
from ..pipelines.ingest import Ingestion


def test_ingestion_returns_dataframe():
    # Mock config.yml content
    mock_config_yaml = """
    data:
      train_path: "mock_train.csv"
      test_path: "mock_test.csv"
    """

    # Create dummy DataFrames to return
    mock_train_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    mock_test_df = pd.DataFrame({"a": [5, 6], "b": [7, 8]})

    with patch("builtins.open", mock_open(read_data=mock_config_yaml)):
        with patch("pandas.read_csv") as mock_read_csv:
            # Configure mock return values
            mock_read_csv.side_effect = [mock_train_df, mock_test_df]

            ingestion = Ingestion()
            train, test = ingestion.load_data()

            # Assertions
            assert not train.empty
            assert not test.empty
            assert train.equals(mock_train_df)
            assert test.equals(mock_test_df)
