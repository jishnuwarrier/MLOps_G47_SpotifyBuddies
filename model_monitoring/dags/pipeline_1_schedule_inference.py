from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import requests
from datetime import datetime


# Function to send inference request to FastAPI
def send_inference_request():
    url = "http://fastapi_server:80/api/playlist/recommend/"
    data = {"user_ids": [0, 1, 2]}
    response = requests.post(url, json=data)
    result = response.json()
    print("Inference result:", result)


# Define the Airflow DAG
dag = DAG(
    "inference_scheduling",
    default_args={
        "owner": "airflow",
        "retries": 3,
    },
    schedule_interval="@hourly",  # Adjust your scheduling as needed
    start_date=datetime(2025, 5, 5),
)

# Task to invoke the FastAPI server
inference_task = PythonOperator(
    task_id="send_inference_request",
    python_callable=send_inference_request,
    dag=dag,
)
