import pickle
import asyncio

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import aiohttp
from datetime import datetime


# Async function to send a single request
async def send_request(session, url, user_batch):
    try:
        async with session.post(url, json={"user_ids": user_batch}) as response:
            result = await response.json()
            print("Inference result:", result)
    except Exception as e:
        print("Request failed:", e)


# Async Function to send all the request
async def send_all_requests(input_data):
    tasks = []
    url = "http://fastapi_server:80/api/playlist/recommend/"

    input_list = sorted(input_data)
    # Create 100 slices (each 100 users)
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(input_list), 1000):
            batch = input_list[i: i + 1000]
            tasks.append(send_request(session, url, batch))

        await asyncio.gather(*tasks)


# Function to send inference request to FastAPI
def send_inference_request():
    with open("/mnt/object/inference_data/selected_user_ids.pkl", "rb") as f:
        input_data = pickle.load(f)

    asyncio.run(send_all_requests(input_data))


# Define the Airflow DAG
dag = DAG(
    "inference_scheduling",
    default_args={
        "owner": "airflow",
        "retries": 3,
    },
    schedule_interval="@daily",
    start_date=datetime(2025, 5, 5),
)

# Task to invoke the FastAPI server
inference_task = PythonOperator(
    task_id="send_inference_request",
    python_callable=send_inference_request,
    dag=dag,
)
