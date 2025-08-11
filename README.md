# MLOPS_pipeline
A minimal end-to-end MLOps pipeline: train, track, package, deploy, and monitor an ML model (Iris or California Housing) using Git, DVC, MLflow, Docker, Flask/FastAPI, and GitHub Actions. Includes experiment tracking, data/model versioning, CI/CD, API deployment, logging, and monitoring following MLOps best practices.
docker pull diliprc96/house_price_predictor:latest

docker network create mlops-net

docker run -d --name house-app --network mlops-net \
  -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=file:///workspaces/MLOPS_pipeline/mlruns \
  -v /workspaces/MLOPS_pipeline/mlruns:/workspaces/MLOPS_pipeline/mlruns \
  diliprc96/house_price_predictor:latest

docker run -d --name prometheus --network mlops-net \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

docker run -d --name grafana -p 3000:3000 grafana/grafana

curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"MedInc": 6, "HouseAge": 25, "AveRooms": 8, "AveBedrms": 1, "Population": 800, "AveOccup": 2.5, "Latitude": 37, "Longitude": -122}'

docker network connect mlops-net prometheus
docker network connect mlops-net grafana

http://prometheus:9090