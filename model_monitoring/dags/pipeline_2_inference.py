import asyncio
import aiohttp
import random
import pickle
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# === CONFIGURATION ===
PKL_PATH = "/path/to/users.pkl"  # Your actual pickle path
ENDPOINT_URL = "http://129.114.25.165:5000/api/playlist/recommend/"
RUN_INTERVAL_MINUTES = 10
USERS_PER_RUN = 100
MAX_DELAY_SECONDS = 480  # 8 minutes delay range
TIMEOUT_BUFFER = 30      # Additional buffer time


# === Load users once ===
def load_users():
    with open(PKL_PATH, "rb") as f:
        return pickle.load(f)

def load_users1():
    users = []
    for i in range(100):
        entry = {}
        user_ids = [random.randint(1, 10), random.randint(1, 10)]
        entry['user_ids'] = user_ids
        users.append(entry)
    return users

# Global variable to hold the loaded users
users = load_users1()


# === Main simulation function ===
def simulate_users_random_timing(**context):
    logger = context["ti"].log
    random.shuffle(users)  # Shuffle once per run to ensure random distribution
    selected_users = users[:USERS_PER_RUN]
    logger.info(f"[{datetime.now()}] Picked {len(selected_users)} users")

    async def send_request(user, session, delay):
        await asyncio.sleep(delay)
        payload = {"user_ids": user['user_ids']}
        try:
            logger.info(f"[{datetime.now()}] Delay={delay:.2f}s Sending: {payload}")
            async with session.post(ENDPOINT_URL, json=payload) as resp:
                resp_text = await resp.text()
                logger.info(f"[{datetime.now()}] Response for user [{resp.status}]: {resp_text}")
        except Exception as e:
            logger.error(f"[{datetime.now()}] Error for user: {e}")

    async def simulate_all():
        async with aiohttp.ClientSession() as session:
            tasks = [
                send_request(user, session, delay=random.uniform(0, MAX_DELAY_SECONDS))
                for user in selected_users
            ]
            try:
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=MAX_DELAY_SECONDS + TIMEOUT_BUFFER)
            except asyncio.TimeoutError:
                logger.error(f"[{datetime.now()}] Timeout: Not all requests completed in time.")

    asyncio.run(simulate_all())
    logger.info(f"[{datetime.now()}] Simulation finished.")


# === DAG definition ===
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 5, 10),
    'retries': 0,
}

dag = DAG(
    dag_id='user_simulation_random_spread',
    default_args=default_args,
    schedule_interval=f'*/{RUN_INTERVAL_MINUTES} * * * *',
    catchup=False,
    max_active_runs=1,
    tags=["simulation"],
)

simulate_task = PythonOperator(
    task_id='simulate_users',
    python_callable=simulate_users_random_timing,
    provide_context=True,
    dag=dag,
)

simulate_task