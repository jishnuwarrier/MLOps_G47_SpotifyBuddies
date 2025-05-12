import asyncio
import aiohttp
import random
import pickle
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# === CONFIGURATION ===
PKL_PATH = "/mnt/object/positives_splits/test_positives.pkl"
ENDPOINT_URL = "http://129.114.25.165:8000/api/playlist/recommend/"
FEEDBACK_URL = "http://129.114.25.165:5000/feedback"
RUN_INTERVAL_MINUTES = 1
USERS_PER_RUN = 25
MAX_DELAY_SECONDS = 30  # 8 minutes delay range
TIMEOUT_BUFFER = 15      # Additional buffer time


# === Load users once ===
def load_users():
    users_dict = {}
    with open(PKL_PATH, "rb") as f:
        users_dict = pickle.load(f)
    user_ids = [user_id for user_id in users_dict]
    return user_ids # list of user_ids

def load_users1():
    users = []
    for i in range(100):
        entry = {}
        user_ids = [random.randint(1, 10), random.randint(1, 10)]
        entry['user_ids'] = user_ids
        users.append(entry)
    return users

# Global variable to hold the loaded users
users = load_users()


# === Main simulation function ===
def simulate_users_random_timing(**context):
    logger = context["ti"].log
    random.shuffle(users)  # Shuffle once per run to ensure random distribution
    selected_users = users[:USERS_PER_RUN]
    logger.info(f"[{datetime.now()}] Picked {len(selected_users)} users")

    async def send_request(user_id, session, delay):
        await asyncio.sleep(delay)
        payload = {"user_ids": [user_id]}
        try:
            logger.info(f"[{datetime.now()}] Delay={delay:.2f}s Sending: {payload}")
            async with session.post(ENDPOINT_URL, json=payload) as resp:
                resp_json = await resp.json()
                logger.info(f"[{datetime.now()}] Response for user [{resp.status}]: {resp_json}")
                return user_id, resp_json
        except Exception as e:
            logger.error(f"[{datetime.now()}] Error for user: {e}")
            return user_id, {}

    async def send_feedback(session, user_id, playlist_ids):
        print(playlist_ids, type(playlist_ids))

        if not playlist_ids or len(playlist_ids) < 2:
            print(f"Insufficient playlists for user {user_id}, skipping feedback.")
            return

        positive = random.choice(playlist_ids)
        ignored = [pid for pid in playlist_ids if pid != positive]

        payload = {
            "user_id" : user_id,
            "like_playlist": positive,
            "other_playlists": ignored
        }

        try:
            async with session.post(FEEDBACK_URL, json=payload) as response:
                resp_text = await response.text()
                print(f"feedback response for user {user_id}: {resp_text}")
        except Exception as e:
            print(f"Error in send_feedback for user {user_id}: {e}")

    async def simulate_all():
        async with aiohttp.ClientSession() as session:
            tasks = [
                send_request(user, session, delay=random.uniform(0, MAX_DELAY_SECONDS))
                for user in selected_users
            ]

            responses = await asyncio.gather(*tasks)
            print(responses)

            feedback_tasks = []
            for user_id, response_json in responses:
                print(user_id, response_json)
                playlist_ids = response_json[0].get('playlists', [])
                feedback_tasks.append(send_feedback(session, user_id, playlist_ids))

            await asyncio.gather(*feedback_tasks)

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