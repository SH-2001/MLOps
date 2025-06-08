import pandas as pd
from ..pipelines.clean import Cleaner


def test_clean_data_removes_columns():
    df = pd.DataFrame(
        {
            "id": [73746, 33999, 246435, 224841, 87273],
            "Gender": ["Male", "Male", "Female", "Female", "Male"],
            "Age": [27, 70, 23, 50, 40],
            "HasDrivingLicense": [1, 1, 1, 1, 1],
            "RegionID": [21, 28, 6, 15, 29],
            "Switch": [None, 1, None, None, 0],
            "VehicleAge": [
                "< 1 Year",
                "1-2 Year",
                "< 1 Year",
                "1-2 Year",
                "1-2 Year",
            ],
            "PastAccident": [None, "No", None, "Yes", "Yes"],
            "AnnualPremium": [
                "£1,510.65",
                "£3,111.50",
                "£2,014.10",
                "£131.50",
                "£1,371.85",
            ],
            "SalesChannelID": [152, 26, 152, 124, 26],
            "DaysSinceCreated": [264, 80, 232, 60, 273],
            "Result": [0, 0, 0, 0, 1],
        }
    )

    cleaner = Cleaner()
    cleaned = cleaner.clean_data(df)

    assert "id" not in cleaned.columns
    assert "SalesChannelID" not in cleaned.columns
    assert "Age" in cleaned.columns
    assert not cleaned["Age"].isnull().any()
