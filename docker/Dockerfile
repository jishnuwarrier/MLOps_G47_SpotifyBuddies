FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get update && apt-get install -y unzip && curl https://rclone.org/install.sh | bash

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt