import pandas as pd
from ..pipelines.clean import Cleaner


def test_clean_data_removes_columns():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "SalesChannelID": [1, 2],
            "VehicleAge": [1, 2],
            "DaysSinceCreated": [10, 20],
            "Gender": ["Male", "Female"],
            "Age": [35, None],
        }
    )

    cleaner = Cleaner()
    cleaned = cleaner.clean_data(df)

    assert "id" not in cleaned.columns
    assert "SalesChannelID" not in cleaned.columns
    assert "Age" in cleaned.columns
    assert not cleaned["Age"].isnull().any()
