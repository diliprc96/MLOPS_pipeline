import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error  
def train_and_log_model(X_train, X_test, Y_train, Y_test, model_class, model_name, params=None):
    """Train model (by class) with given params, log to MLflow as a run, return RMSE and run_id."""
    if params is None:
        params = {}

    with mlflow.start_run() as run:
        model = model_class(**params)
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)
        rmse = root_mean_squared_error(Y_test, preds)

        # Log parameters and metrics
        mlflow.log_param("model_type", model_name)
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("rmse", rmse)
        signature = infer_signature(X_test, preds)
        mlflow.sklearn.log_model(
            model, "model", 
            signature=signature,
            input_example=X_test.iloc[:1]
        )
        print(f"{model_name} ({params}) trained with RMSE: {rmse}")
        return {"run_id": run.info.run_id, "rmse": rmse, "model_type": model_name, "params": params}

def train_multiple_models(csv_path, model_configs, seeds):
    """Train and log runs for all models/configs and all seeds. Return list of run info dicts."""
    df = pd.read_csv(csv_path)
    X = df.drop(columns="MedHouseVal")
    Y = df["MedHouseVal"]
    mlflow.set_experiment("california_housing_regression")
    run_infos = []

    for config in model_configs:
        model_class = config["model_class"]
        model_name = config.get("model_name", model_class.__name__)
        model_params = config.get("params", {})
        for seed in seeds:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
            params = dict(model_params)  # copy
            if "random_state" in model_class().get_params():
                params["random_state"] = seed
            run_info = train_and_log_model(X_train, X_test, Y_train, Y_test, model_class, model_name, params)
            run_infos.append(run_info)
    return run_infos
