from ..pipelines.ingest import Ingestion


def test_ingestion_returns_dataframe():
    ingestion = Ingestion()
    train, test = ingestion.load_data()
    assert not train.empty
    assert not test.empty
