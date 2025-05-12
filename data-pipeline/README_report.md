## Persistent Storage

### Object Store in CHI@TACC

`object-persist-project47`
`~7GB used`
[see on chi@tacc](https://chi.tacc.chameleoncloud.org/project/containers/container/object-persist-project47)

Contains:

1. triplets_data - contains the PyTorch tensor file containing training data for playlist ranker model
2. positives_splits -
3. val_eval_batches -
4. user_pairing_model
5. other_utils

### Block Store in KVM@TACC

`block-persist-project47`
`Size 50GB allocated`
[see on kvm@tacc](https://kvm.tacc.chameleoncloud.org/project/volumes/42b1865a-6ff9-44ed-9f8d-8ca639f4b97c/)

Contains data for:

1. Postgres(used for storing feedback data, Metabase dashboard metadata, Grafana and Prometheus related data)
2. Grafana
3. Prometheus
4. MinIO

## Offline data

> Needs some input from Agustin

## Data Pipeline

## Data Dashboard

We use a metabase dashboard to show the metrics related to data and data quality. For the training data, we show metrics like distinct users, average blocks per user etc. For ensuring data quality of the recommendation system, we need to make sure that for a user, we have 70%-20%-10% split of the positive playlist in train, validation and test dataset respectively. For this reason, we display metrics like

### Data Leakage

## Online Data

User interaction with the recommendation system is done via an Airflow script. The script uses the test-data split stored in the object store. The Airflow job is run every 10 minutes - it selects a particular amount of users randomly and calls the fastapi `/recommed` api over the duration of 10 minutes to get the recommendations. We offer 5 playlists recommendations per api call.

Regarding closing the loop, we take feedback from the user on the application. The UI for this application is hosted. Currently, we only support the user liking 1 playlist out of the 5 playlists. For simulation purposes though, we call a fastapi endpoint called `/feedback` through the same airflow script. This stores the user feedback in the postgres database.

The request and response format as given below:

**/recommend API**

Request

```
curl -X 'POST' \
  'http://<url>/api/playlist/recommend/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_ids": [
    22434
  ]
}'
```

Response

```
[
  {
    "user_id": 22434,
    "playlists": [
      14546, 28783, 209745, 29747, 32649
    ]
  }
]
```

**/feedback API**

Request

```
curl -X 'POST' \
  'http://<url>/feedback' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_id": 22434,
  "like_playlist": 28783,
  "other_playlists": [
    14546, 209745, 29747, 32649
  ]
}'
```

Response

```
{
  "msg": "Feedback Sent"
}
```

## Data for Retraining

The data from Postgres is extracted using another Airflow script on a weekly basis. Data is split into train, validation and test and converted into the required format for retraining. `rclone` is used inside the Airflow script using python subprocesses to push the new data into the object store. This data will be appended to the main training data and used for training. In recommendation systems, usually we deal with a sliding window style training data. So as we collect more and more data, ideally we will start removing data from the beginning to keep the data size reasonalble. This also make sense for a recommedation system as people are much more likely to like a recommendation based on their interest during the past month than say a year.
