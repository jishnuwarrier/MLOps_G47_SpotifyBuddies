from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import psycopg2
import pickle
import os
import subprocess

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id='fetch_postgres_data_process',
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
        dbname='mydb',
        user='user',
        password='password'
    )
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM prod_data;')
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    print("Got all the rows")

    pickle_file_path = '/tmp/prod_data.pkl'  # Pickle file location (inside the container)

    with open(pickle_file_path, 'wb') as f:
        pickle.dump(rows, f)

    rclone_command = [
        'rclone', 'copy', pickle_file_path, 'chi_tacc:object-persist-project47',
        '--progress', '--transfers=32', '--checkers=16', '--multi-thread-streams=4', '--fast-list'
    ]

    # Run the rclone command
    result = subprocess.run(rclone_command, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(f"Rclone upload failed with error: {result.stderr}")

    print(f"Rclone upload successful: {result.stdout}")

task_fetch = PythonOperator(
    task_id='fetch_and_process_rows',
    python_callable=fetch_and_process_rows,
    dag=dag
)