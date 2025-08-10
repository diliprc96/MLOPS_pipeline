import mlflow
from mlflow.tracking import MlflowClient

def register_best_model(
        experiment_name="california_housing_regression",
        model_path="model",
        registered_model_name="CaliforniaHousingModel"
    ):
    """
    Finds the best model run by lowest RMSE and registers its model artifact to the Model Registry.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    # Get all runs in this experiment
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )
    if runs_df.empty:
        raise RuntimeError("No runs found. Please train models first.")

    # Sort runs by RMSE ascending
    best_run = runs_df.sort_values("metrics.rmse").iloc[0]
    print(f"Best run_id: {best_run.run_id}, RMSE: {best_run['metrics.rmse']}, model: {best_run['params.model_type']}, params: { {k: v for k,v in best_run.items() if k.startswith('params.') } }")

    client = MlflowClient()

    # Create registered model if not exists
    existing_models = [rm.name for rm in client.search_registered_models()]
    if registered_model_name not in existing_models:
        client.create_registered_model(registered_model_name)

    # Register model version from best run
    model_version = client.create_model_version(
        name=registered_model_name,
        source=f"runs:/{best_run.run_id}/{model_path}",
        run_id=best_run.run_id
    )
    print(f"Model registered: {model_version.name} version {model_version.version}")


    client.transition_model_version_stage(
        name=registered_model_name,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True  # archives previous prod versions!
    )
    print(f"Model version {model_version.version} transitioned to Production.")

    return model_version

if __name__ == "__main__":
    register_best_model()
