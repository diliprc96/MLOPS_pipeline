# from scripts.train_model import model_train
# from scripts.register_best_model import register_best_model

# data_path = "data/california_housing.csv"
# param_grid = [34, 85, 123, 42, 94, 75]  

# # Step 1: Train multiple models and get run info
# run_infos = model_train(data_path, param_grid)

# # Step 2: Select and register best model based on metric
# register_best_model(run_infos, registered_model_name="CaliforniaHousingModel")



from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from scripts.train_model import train_multiple_models
from scripts.register_best_model import register_best_model

# Define models and params
model_configs = [
    {"model_class": LinearRegression, "model_name": "LinearRegression", "params": {}},
    {"model_class": DecisionTreeRegressor, "model_name": "DecisionTreeRegressor", "params": {"max_depth": 5}},
    {"model_class": DecisionTreeRegressor, "model_name": "DecisionTreeRegressor", "params": {"max_depth": 10}},
]

# Seeds to train with
seeds = [42, 85]

data_path = "data/california_housing.csv"

# Train all models on all seeds
run_infos = train_multiple_models(data_path, model_configs, seeds)
print("All runs completed:", run_infos)
register_best_model()