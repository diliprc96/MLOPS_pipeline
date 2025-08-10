FROM python:3.9-slim

WORKDIR /app


COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


COPY api/ ./api/


RUN mkdir -p logs

ENV MLFLOW_TRACKING_URI http://host.docker.internal:5000

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
