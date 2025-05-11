from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import psycopg2
import pickle
import os
import subprocess
import random
from collections import defaultdict

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 0,
    # 'retry_delay': timedelta(minutes=5),
}

print("Randon integer", random.randint(1, 10))

dag = DAG(
    dag_id='retraining_data_fetch',
    default_args=default_args,
    description='Fetch and print rows from postgres weekly',
    schedule_interval='@weekly',
    catchup=False,
    tags=['postgres'],
)

def fetch_and_process_rows():
    conn = psycopg2.connect(
        host='postgres',           # Docker Compose service name for postgres
        port='5432',
        dbname='client',
        user='postgres',
        password='1234'
    )
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM user_feedback;')
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    print("Got all the rows")

    user_groups = defaultdict(list)
    for created_at, user_id, playlist_id, score in rows:
        print(created_at, user_id, playlist_id, score)
        user_groups[(created_at, user_id)].append((playlist_id, score))

    user_to_posneg = defaultdict(list)

    for (created_at, user_id), items in user_groups.items():
        pos_playlist = None
        neg_playlists = []
        for playlist_id, score in items:
            if score == 1:
                pos_playlist = playlist_id
            else:
                neg_playlists.append(playlist_id)
        if pos_playlist:
            user_to_posneg[user_id].append((pos_playlist, neg_playlists))

    train, val, test = [], [], []

    for user_id, posneg_list in user_to_posneg.items():
        print(posneg_list)
        random.shuffle(posneg_list)
        total = len(posneg_list)
        train_end = int(total * 0.7)
        val_end = train_end + int(total * 0.2)

        train_data = posneg_list[:train_end]
        val_data = posneg_list[train_end:val_end]
        test_data = posneg_list[val_end:]

        for pos_playlist, negs in train_data:
            for neg in negs:
                train.append((user_id, pos_playlist, neg))

        for pos_playlist, negs in val_data:
            for neg in negs:
                val.append((user_id, pos_playlist, neg))

        for pos_playlist, negs in test_data:
            for neg in negs:
                test.append((user_id, pos_playlist, neg))

    training_pkl_path = "/tmp/training.pkl"
    validation_pkl_path = "/tmp/validation.pkl"
    testing_pkl_path = "/tmp/testing.pkl"

    with open(training_pkl_path, "wb") as f:
        pickle.dump(train, f)

    with open(validation_pkl_path, "wb") as f:
        pickle.dump(val, f)

    with open(testing_pkl_path, "wb") as f:
        pickle.dump(test, f)

    now = datetime.now()
    date_string = 'chi_tacc:object-persist-project47/' + 'data_'  + now.strftime("%Y-%m-%d")

    rclone_command1 = [
        'rclone', 'copy', training_pkl_path, date_string,
        '--progress', '--transfers=32', '--checkers=16', '--multi-thread-streams=4', '--fast-list'
    ]

    rclone_command2 = [
        'rclone', 'copy', validation_pkl_path, date_string,
        '--progress', '--transfers=32', '--checkers=16', '--multi-thread-streams=4', '--fast-list'
    ]

    rclone_command3 = [
        'rclone', 'copy', testing_pkl_path, date_string,
        '--progress', '--transfers=32', '--checkers=16', '--multi-thread-streams=4', '--fast-list'
    ]

    # Run the rclone command
    result1 = subprocess.run(rclone_command1, capture_output=True, text=True)

    if result1.returncode != 0:
        raise Exception(f"Rclone upload failed with error: {result1.stderr}")

    result2 = subprocess.run(rclone_command2, capture_output=True, text=True)

    if result2.returncode != 0:
        raise Exception(f"Rclone upload failed with error: {result2.stderr}")

    result3 = subprocess.run(rclone_command3, capture_output=True, text=True)

    if result3.returncode != 0:
        raise Exception(f"Rclone upload failed with error: {result3.stderr}")

    print(f"Rclone upload successful")

task_fetch = PythonOperator(
    task_id='fetch_retraining_data',
    python_callable=fetch_and_process_rows,
    dag=dag
)