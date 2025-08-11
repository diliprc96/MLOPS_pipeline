# MLOPS_pipeline
A minimal end-to-end MLOps pipeline: train, track, package, deploy, and monitor an ML model (Iris or California Housing) using Git, DVC, MLflow, Docker, Flask/FastAPI, and GitHub Actions. Includes experiment tracking, data/model versioning, CI/CD, API deployment, logging, and monitoring following MLOps best practices.
docker pull diliprc96/house_price_predictor:latest

docker run -d --name prometheus   -p 9090:9090   -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml   prom/prometheus   --config.file=/etc/prometheus/prometheus.yml   --web.listen-address="0.0.0.0:9090"
docker run -d --name grafana -p 3000:3000 grafana/grafana