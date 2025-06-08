import pandas as pd
from ..pipelines.train import Trainer


def test_pipeline_training():
    trainer = Trainer()

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

    X, y = trainer.feature_target_separator(df)
    trainer.train_model(X, y)

    pred = trainer.pipeline.predict(X)
    assert len(pred) == len(y)
