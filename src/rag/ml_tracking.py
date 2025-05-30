import mlflow
import os
import json

def start_experiment(name="RuleBookAssistant"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(name)

def log_pipeline_params(params: dict):
    for key, value in params.items():
        mlflow.log_param(key, value)

def log_pipeline_metrics(metrics: dict):
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

def log_artifacts(response: str, context: str, completion_token_details: str):
    with open("temp_response.txt", "w") as f:
        f.write(response)
    with open("temp_context.txt", "w") as f:
        f.write(context)
    with open("completion_tokens_detsils.json", "w") as f:
        json.dump(completion_token_details, f)

    mlflow.log_artifact("temp_response.txt")
    mlflow.log_artifact("temp_context.txt")
    mlflow.log_artifact("completion_tokens_detsils.json")
    os.remove("temp_response.txt")
    os.remove("temp_context.txt")
    os.remove("completion_tokens_detsils.json")
