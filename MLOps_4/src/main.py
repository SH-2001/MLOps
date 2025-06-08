import logging
import yaml
import mlflow
import mlflow.sklearn
from pipelines.ingest import Ingestion
from pipelines.clean import Cleaner
from pipelines.train import Trainer
from pipelines.predict import Predictor
from sklearn.metrics import classification_report

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s"
)


def mlflow_main():
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    mlflow.set_experiment("Model Training Experiment")

    with mlflow.start_run() as run:
        # Load data
        ingestion = Ingestion()
        train, test = ingestion.load_data()
        logging.info("Data ingestion completed successfully")

        # Clean data
        cleaner = Cleaner()
        train_data = cleaner.clean_data(train)
        test_data = cleaner.clean_data(test)
        logging.info("Data cleaning completed successfully")

        # Prepare and train model
        trainer = Trainer()
        X_train, y_train = trainer.feature_target_separator(train_data)
        trainer.train_model(X_train, y_train)
        #trainer.save_model()
        logging.info("Model training completed successfully")

        # Evaluate model
        predictor = Predictor()
        X_test, y_test = predictor.feature_target_separator(test_data)
        acc, class_report, roc_auc_score = predictor.evaluate_model(
            X_test, y_test
        )
        report = classification_report(
            y_test, trainer.pipeline.predict(X_test), output_dict=True
        )
        logging.info("Model evaluation completed successfully")

        # Tags
        mlflow.set_tag(
            "preprocessing",
            "OneHotEncoder, Standard Scaler, and MinMax Scaler",
        )

        # Inferring the input signature
        signature = mlflow.models.infer_signature(
            model_input=X_train, model_output=trainer.pipeline.predict(X_test)
        )

        client = mlflow.MlflowClient()

        # Compare against the best previous ROC AUC
        ex_id = mlflow.get_experiment_by_name(
            "Model Training Experiment"
        ).experiment_id
        previous_best = client.search_runs(
            experiment_ids=[ex_id],
            order_by=["metrics.roc DESC"],
            max_results=1,
        )

        if previous_best and len(previous_best) > 0:
            best_prev_roc = previous_best[0].data.metrics["roc"]
            logging.info(f"Best previous ROC AUC: {best_prev_roc:.4f}")
        else:
            best_prev_roc = -1
            logging.info("No previous model found, treating as first run.")

        # Log metrics
        model_params = config["model"]["params"]
        mlflow.log_params(model_params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc", roc_auc_score)
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.sklearn.log_model(
            trainer.pipeline, "model", signature=signature
        )

        # Register the model
        if roc_auc_score > best_prev_roc:
            logging.info(
                "New model outperforms previous best. Registering model."
            )
            model_name = "insurance_model"
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri, model_name)
        else:
            logging.info(
                "Model is not better than previous. Skipping registration."
            )

        logging.info("MLflow tracking completed successfully")

        # Print evaluation results
        print("\n============= Model Evaluation Results ==============")
        print(f"Model: {trainer.model_name}")
        print(f"Accuracy Score: {acc:.4f}, ROC AUC Score: {roc_auc_score:.4f}")
        print(f"\n{class_report}")
        print("=====================================================\n")


if __name__ == "__main__":
    mlflow_main()
