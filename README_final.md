# README

# “Spotify Buddies”: an organic playlist recommender

---

## Overview

- We’ll design a machine learning system for organic playlist recommendations to be integrated in music streaming services, specifically Spotify.
- The main feature of the model is that it will recommend ‘organic’ playlists to the user, meaning a playlist that was created by a human user with similar music taste to the requesting user, as opposed to an algorithmically-generated synthetic playlist.

## Value Proposition

- We strongly believe that music streaming services are missing a huge opportunity in no leveraging a social dimension in their products.
- Organic playlists recommendations is one way to incorporate such social aspect, encouraging users to engage with the playlists from other users, and sharing their music taste with each other.
- This might improve business metrics related to user engagement, such as screen time, hours of music play, size of user music library, and general retention of paid-subscription members.

### Contributors

| Name | Responsible for | Link to their commits in this repo |
| --- | --- | --- |
| All team members | Idea, value proposition, basic setup of ML problem |  |
| Agustin Leon | Model training, experiment tracking (Units 4 & 5)  | [Link](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/commits/main?author=AguLeon) |
| Akhil Manoj | Data pipeline (unit 8) | [Link](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/commits/main?author=AkM-2018)  |
| Anup Raj Niroula | Model serving and monitoring (Units 6 & 7) | [Link](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/commits/main?author=ARNiroula) |
| Jishnu Warrier | Continuous pipeline (unit 3) | [Link](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/commits/main?author=jishnuwarrier) |

### System diagram

![mlops-presentation.jpg](./mlops-presentation.jpg)

### Summary of outside materials

|  | How it was created | Conditions of use |
| --- | --- | --- |
| Million Playlist Dataset | Created by Spotify for the RecSys Challenge 2018 | Available for research open-ended use |
| EchoNest Taste Profile Subset | Created by LabROSA @ Columbia University, currently maintained by Spotify | Available for open-ended use |

### Summary of infrastructure requirements

| Requirement | How many/when | Justification |
| --- | --- | --- |
| `m1.medium` VMs | 4 for entire project duration | Hosts dashboard, model API, ML Flow |
| NVIDIA `gpu_A100` | 4 hour block 2-3 times a week | Required for training tasks |
| Floating IPs | 1 for entire project duration, 1 for sporadic use | Required for endpoints using FastAPI |
| Persistent volume | 100 GB | Storage for datasets, model artifacts, logs, batched inferences |

### Summary of extra difficulty points that we’ll aim for

| Unit | Topic | Difficulty point | Status |
| --- | --- | --- |
| 1 | ML Ops Intro Unit | Using multiple models (we’ll be using two models) | Completed |
| 5 | Model training infrastructure and platform | Ray Tune | Completed |
| 6 | Model serving and monitoring | Develop multiple options for serving | Semi-completed |
| 7 | Evaluation and monitoring  | Monitor for model degradation | Completed |
| 8 | Data Pipeline | Interactive Data Dashboard | Completed |

# Detailed design plan

# Model Training & Infrastructure (Agustin)

## General overview

This pipeline uses two models, a K-Nearest Neighbors model (KNN) for pairing users and a Bayesian Personalized Ranking Model for ranking playlists (BPR).

The pipeline is the following:

1.  A user_id requests recommendations
2. We find itsneighbors based on the pre-computed KNN model pairings
3. We get the list of playlists that belong to their neighbors
4. We sample those playlists (also pre-computed and fixed)
5. We pass the user_id and the list of playlists onto the trained BPR model
6. We get the top_K ranked playlists as a recommendation (we use top_K = 5)

In a real pipeline computing pairings (running the KNN model) would happen much less frequently than re-training the BPR model.

The KNN Model takes about 2 hours to produce pairings.

The BPR Model training time depends a lot on some parameters, but it broadly takes 1hr30mins for 5 epochs.

For the BPR Model we trained implementing MLFlow and Ray Tune.

We will cover these models in detail below, in addition to the training process for both of them.

---

---

## 1) KNN Model Description and ‘Training’

- Relevant file:

[KNN_script](training/spotifybuddies_userpairingmodel_v0.py)

- The KNN Model calculates neighbors using cosine similarity among the vector of liked songs that each user has (this comes from the ENTP dataset).
- Benefits of KNN are its simplicity and interpretability, and it’s a good baseline model for getting pairings.
- Before running the KNN model we first need to answer a few questions and make some decisions:
    1. How many neighbors per user?
    2. Do we enforce a minimum and/or a max?
- For this, we choose to decide not only based on the amount of neighbors, but also, on the amount of playlists that the given user has access to, through its neighbors.
- We ask the KNN model to find neighbors at least until the user has at least 20 playlists owned by its neighbors (users that are very unique will have to be paired to more disimilar users to match this requirement). We also set the max of user neighbors at 50.
- Below we can see some statistics and metrics. On average each user gets 37 neighbors, median is 39 neighbors. Min and max are 1 and 100 respectively. Users that have 1 neighbor might have reached the 100 playlists requirement with just one neighbor.
- 

<aside>
📊

 Neighbor Count Statistics:

- Users analyzed : 101,713
- Min neighbors : 0
- Max neighbors : 50
- Mean neighbors : 36.79
- Median neighbors : 39.0
- 25th percentile : 30.0
- 75th percentile : 48.0
- 99th percentile : 50.0
</aside>

![Distribution of Neighbor Count per User](training/images_log/image.png)

- We also explore ‘user popularity’. We exclude users that were never selected as a neighbor. The median is 57, and on average each user is the neighbor of 94 other users. Some users are extremely popular, with over 600 selections.

<aside>
📊


User Popularity (Times a User Was Paired as Neighbor):

- Users selected as neighbor: 39,634
- Min times selected : 1
- Max times selected : 2348
- Mean : 94.41
- Median : 57.0
- 90th percentile : 207.0
- 99th percentile : 609.0
</aside>

![Distribution of User Popularity](training/images_log/image_1.png)

- Finally we check the playlist availability through neighbors. We can see a pretty soft shape, median is 222 playlists. Note that some users will have access to over 400 playlists. This could delay inference for that user. We take care of that next.

<aside>
📊

Playlists Accessible via Neighbors:

- Users analyzed : 101,713
- Min playlists : 0
- Max playlists : 1248
- Mean playlists : 266.27
- Median playlists : 221.0
- 25th percentile : 122.0
- 75th percentile : 372.0
- 99th percentile : 841.0
</aside>

![Distribution of Playlist Access via Neighbors](training/images_log/image_2.png)

### Final step: create dictionary that quickly maps user_id to a list of playlist id’s belonging to its neighbors. This is used for inference.

- For each user, we collect all the playlists owned by its neighbors.
- Since sometimes we get over 400 playlists for a user, we want to standardize this and get a similar amount of playlists for every user.
- We will sample randomly 75 users for each user (if they have less than 75 we will just take whatever much they have).
- In production the KNN would be recomputed with some frequency and we would sample different playlists each time. This could give the model some ‘freshness’, in which users don’t always get recommendations from the same set of playlists.

---

---

## 2) BPR Model Description and Training

- We decided to implement a Bayesian Personalized Ranking model (BRP) for our pipeline because of its effectiveness and simplicity. It also trains reasonably fast and it’s lightweight, so it’s suited for our timeline. A full model checkpoint of this model weighs about 1GB.

This model learns user and playlist embeddings, that will capture both user’s similarity and playlist’s similarity. The model is trained on triplets (user_id, positive_playlist_id, negative_playlist_id), where each triplet reveals a relative preference for a user for one playlist over some other playlist.

- Negative playlists aren’t necessarily disliked. The positive was just preferred over the negative playlist. A negative playlist might perfectly become a positive if it’s recommended among another set of playlists.

### Model Structure

- We will have both user embeddings and playlist embeddings. Our final embeddings have a dimension of 128.
- BPR will compute a dot product between the embeddings, this is the score:

$$
\text{score}*{u,i} = \mathbf{u}^\top \cdot \mathbf{i}, \quad \text{score}*{u,j} = \mathbf{u}^\top \cdot \mathbf{j}
$$

- u: embedding vector for user uuu
- i: embedding vector for positive playlist iii
- j: embedding vector for negative playlist jjj
- score(u,i): predicted preference score of user uuu for playlist iii

- The loss function is given by:

$$
\mathcal{L}*{\text{BPR}} = -\frac{1}{N} \sum*{(u,i,j)} \log \left( \sigma(\text{score}*{u,i} - \text{score}*{u,j}) \right)
$$

Where:

- sigma is the sigmoid function.
- (u,i,j) are user, positive playlist, and negative playlist triplets.

### Model Evaluation

For each validation positive sample:

- The model ranks 1 positive playlist among 50 negatives.
- We compute:
    - **MRR (Mean Reciprocal Rank)**: Average inverse rank of the true positive.
    - **Hit@K**: Measures if the true positive is in the top K results.

### Data Leakage?

- We took measures to protect the implementation from data leakage, as explained in the data preprocessing.
- The main precaution is that the model is evaluated with unseen playlists by the user. This replicates the “past vs future” nature of recommender systems. The model was trained on the “past”, and we know whether the user liked or not the users we recommended. In the future (evaluation) we won’t offer those playlists he already expressed a preference for. This is how recommender works: once you get a playlist and you add it to your collection, you should not get it as a recommendation again.

### Final parameters:

| **Parameter** | **Value** | **Description** |
| --- | --- | --- |
| `embedding_dim` | 128 | Size of user and playlist embedding vectors |
| `batch_size` | 16,384 | Number of triplets per training batch |
| `learning_rate` | 0.005 | Step size for the optimizer |
| `epochs` | 5 | Number of training epochs |
| `NUM_NEGATIVES` | 4 | Number of negative playlists sampled per positive in each training triplet |

---

---

## 3) BPR Model Training Jobs with MLFlow and Ray Tune

[BPR_script](training/spotifyBuddies_train_mlflow_v3.py)

- For training the BPR Model we implemented a training script that incorprated MLFlow and Ray Tune. We have two main runs: one with MLFlow only, and then another using Ray Tune.

### MLFlow Implementation:

- Below some code snippets from the training script that take care of setting MLFlow
- We used tags such as ‘full’ and ‘toy’, depending on which dataset was being used. Also we grouped jobs in different experiments.

```bash
MLFLOW_EXPERIMENT_NAME = 'SpotifyBuddies_experiment5_RayTune'
MLFLOW_TAGS = {
    "platform": "chameleon_cloud",
    "mode": "full",
    "run_type": "baseline"
}
```

- We directly put the tracking URI to our KVM@@TACC node hosting MLFlow and MinIO

```bash
mlflow.set_tracking_uri("http://129.114.27.215:8000")
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
mlflow.start_run()
mlflow.log_params(config)
mlflow.set_tags(MLFLOW_TAGS)
```

- We are sending to our MLFlow our custom metrics: MRR and Hit@K

```bash
mlflow.log_metrics({
    "train_loss": avg_train_loss,
    "val_mrr": val_mrr,
    **{f"hit-{k}": val_hit_rates[k] for k in val_hit_rates}
}, step=epoch)
```

- Finally we log into ML Flow our artifacts:

```bash
mlflow.pytorch.log_model(model, artifact_path="final_model")
mlflow.log_artifact("last_run_id.txt")
mlflow.log_artifact("training_config.json")
mlflow.log_artifact(MODEL_CONFIG_PATH)
mlflow.log_artifacts(checkpoint_bundle_dir, artifact_path="checkpoints")
```

### Ray Tune Implementation

- Ray Tune is used to **automate hyperparameter tuning** by launching multiple trials in parallel, each with a different configuration. It works together with MLflow to track results.
- We set yo the following exploring space:

```bash
RAY_SEARCH_SPACE = {
    "embedding_dim": tune.choice([64, 128]),
    "learning_rate": tune.loguniform(1e-4, 1e-2),
    "batch_size": tune.choice([8192, 16384]),
    "val_batch_dir": VAL_EVAL_BATCH_DIR,
    "triplet_path": TRIPLET_DATASET_PATH
}
```

- Then we put the config inside the training function:

```bash
report(metrics={
    "train_loss": avg_train_loss,
    "val_mrr": val_mrr,
    **{f"hit@{k}": val_hit_rates[k] for k in val_hit_rates}
})
```

- And we launch Ray Tune using:

```bash
tune.run(
    train_with_resources,
    config=RAY_SEARCH_SPACE,
    metric="val_mrr",
    mode="max",
    num_samples=RAY_NUM_SAMPLES,
    name="bpr_hpo",
    storage_path="~/ray_results",
    resume="AUTO"
)
```

- We only did one Ray Tune run on the full dataset, that created four experiments on MLFlow.

---

---

## 4) Main Results

- Our main run achieved on the validation set a MRR of 0.503 and the following Hit@K values:
    - Hit@1 = 0.33
    - Hit@5 = 0.72
    - Hit@10 = 0.90

![MRR across 5 epochs](training/images_log/image_3.png)

- Hit@5 is a very important metric because our model serving service will recommend 5 playlists to users. We can see that in 72% of cases a positive playlist existing in a bag with 50 negative playlists, will be ranked inside the top 5 playlists.

![Hit@5 across 5 epochs](training/images_log/image_4.png)

### Ray Tune results

- With respect to Ray Tune, we are including below a comparison of the experiments that Ray Tune handled. Since we ran Ray Tune just one time on the full dataset, we were able to explore only variations on the learning rate.

![Comparison of learning rates for Ray Tune experiments](training/images_log/image_5.png)

- We can see that Ray Tune identified some space for improvement, but it was very minor. Probably we need to experiment further in order to discover more significant improvements. Basically we tried too little of variations of learning rates, that allowed to increase MRR from 0.47889 to 0.47912, which is basically negligible.

![Comparison of metrics (MRR and Hit@5) for Ray Tune experiments](training/images_log/image_6.png)

- Probably it would have been more interesting to explore another hyperparameter, such as embedding dimensions of the BPR Model, but still Ray Tune proved to be a valuable tool at exploring hyperspace optimization in a consistent way.


## Model serving and monitoring platforms (Anup)

- **Model Serving**
    - **Serving from an API endpoint**
        - Will use FastAPI to expose API endpoints to get model inferences. Since FastAPI supports native async functions, we will use asynchronous functions for the API endpoints
    - **Identify requirements**
        - Our model will give out inferences/recommendation in a fixed schedule (similar to Spotify’s `Top Picks for Today` features)
            - We will do experiment with inferences using dynamic batching with and without delay. We picked these since we are targeting for inferences in a schedule
        - Will be completely cloud-native. Our primary focus will be to serve concurrent user requests.
    - **Model optimizations to satisfy requirements**
        - We are aiming for high accuracy, so we will focus on optimization that makes the model more efficient but won’t affect its accuracy like graph optimization.
    - **System optimizations to satisfy requirements**
        - When taking into account the worst case scenario, it is possible that all the active users will request the playlist recommendation concurrently, which our project should be able to handle in optimized manner.
            - Our assertion is that user's will get the recommendation in a fixed scheduler. So, we will explore the possibility of using CRON jobs and Celery Queueing service to manage the schedule and asynchronous queues to handle the request for inferences.
    - **Develop multiple options for serving**
        - We will be benchmarking deploying in Server-Grade GPU, Server-Grade CPU. We won't be deploying in on-device for this project so we won't take that into consideration.
            - We will produce a comparative report that will include the overall throughput, costs, on the basis of which we will determine which system to deploy the model during production.
- **Model Monitoring**
    - **Offline evaluation of model**
        - Model Evaluation
            - We will use the similarity metrics defined in the model training section to evaluate the model based on the similarity with the user’s own playlist and the recommended playlist
            - Test on known failure modes
                - For Cold Start Users i.e. Users with no generated, followed, liked playlist (Completely New Users): We will simply show the most popular/most liked/ most followed playlist to the users.
            - Unit tests
                - Our current plan is to add the unit test of the models to a web-hook, so that when-ever we modify the model code and try to push the updated component, it needs to pass the unit-test first
    - **Load test in staging**
        - We will load test the service by increasing the total number of concurrent users request. We will include it in the CI/CD pipeline and log the results after every deployment to staging area for reporting and further references
    - **Online evaluation in canary**
        - We will act as different types of Spotify users such as:
            - Users with no generated, followed, liked playlist (Cold start users)
                - At first, we will recommend the overall most popular playlist i.e. playlist with most follow/likes
                - Then, we will at first track the user’s activity so that we can build up a sort of User Persona. Only after that, will we apply the personalized recommendation
            - Users with all of their playlist
                - We will evaluate recommendation based 2 factors
                    - Similarity to the playlist that generated, liked, followed have as the users
                    - Our own subjective evaluation of the playlist taste
    - **Closing the feedback loop**
        - Business-specific evaluation
            - Our business-specify evaluation will be the one that is currently used by Spotify, which is most probably (since we don’t know all the internal business evaluation metrics Spotify uses) something along the lines of Time Spent Listening, Click-through rates
                - All the different metrics will feed into the overall North Start Metrics of Spotify, which is the time spent listening to music
        - We will primarily track user’s activity and engagement metrics to get the feedback about the quality of our model. The activity will consists of
            - Explicit liking and saving the recommended playlists
            - Implicitly activity such as listing to the more than some arbitrary percentages (Will be iteratively modified based on the business metrics and stakeholder decision) of the songs in the playlist
    - **Monitor for model degradation**
        - We will employ Prometheus service to check for the degradation of model’s performance. Our current idea is that if the user activity stops improving/increasing even after getting the model, then we can assume that the model’s performance has degraded. Then, we will trigger the automated machine training pipeline

# Data pipeline (Akhil)

## Persistent Storage

### Object Store in CHI@TACC

`object-persist-project47`
`~7GB used`
[see on chi@tacc](https://chi.tacc.chameleoncloud.org/project/containers/container/object-persist-project47)

Contains:

1. **triplets_data** - Contains the PyTorch tensor file containing training data
2. **val_eval_batches** - For evaluation
3. **positives_splits**
4. **user_pairing_model**
5. **other_utils**

### Block Store in KVM@TACC

`block-persist-project47`
`Size 50GB allocated`
[see on kvm@tacc](https://kvm.tacc.chameleoncloud.org/project/volumes/42b1865a-6ff9-44ed-9f8d-8ca639f4b97c/)

Contains data for:

1. **Postgres**(used for storing feedback data, Metabase dashboard metadata, Grafana and Prometheus related data)
2. **Grafana**
3. **Prometheus**
4. **MinIO**

**Scripts for persistent storage setup**

- [Provision object store on Chameleon](./provision_object_storage.sh)
- [Provision block store on Chameleon](./provision_block_storage.sh)
- [Setup object store on the instance](./setup_object_store.sh)
- [Setup block store on the instance](./setup_block_store.sh)

## Offline data

Covered below in detail.


## Data Dashboard

We use a Metabase dashboard to monitor various metrics related to both the data and its quality.

For the training data, we display key metrics such as the number of distinct users and the average number of user-blocks per user.
A `user-block` is defined as a triplet: `(user_id, playlist_id_positive, playlist_id_negative_list)`. Each user-block groups together entries by user_id and their preferred `playlist_id_positive`, along with a list of `playlist_id_negative` values. These negative playlist IDs represent alternatives that the user preferred less than the positive one.
To ensure the integrity and consistency of our training data, we validate that each `user-block` contains only one `playlist_id_positive` per `(user_id, playlist_id_positive)` combination.

![training-widgets](./data-pipeline/images/training.png)

For feedback data, we track metrics such as:

- Daily and weekly user counts providing feedback
- Average feedback score per day, to identify potential recommendation bias
- Invalid user-blocks, similar to training data checks, to ensure data integrity
- Feedback volume over time, which helps determine when enough new data has accumulated to trigger model retraining
- Cumulative feedback growth, providing insight into overall dataset expansion
- Number of distinct feedback users, which helps us monitor user diversity and avoid over-representation by "power users"

![feedback-widgets](./data-pipeline/images/feedback.png)

## Online Data

[Airflow script for user simulation](./../model_monitoring/dags/pipeline_2_inference.py)

User interaction with the recommendation system is simulated using an **Airflow script**. This script utilizes the **test data split** stored in an object store and runs as a scheduled job every **10 minutes**. In each run, it randomly selects a subset of users and invokes the FastAPI `/recommend` endpoint over a 10-minute window to fetch playlist recommendations. Each API call returns **5 playlist recommendations** for a given user.

To **close the feedback loop**, we collect user responses through a hosted UI. Currently, the interface allows the user to **like one playlist out of the five recommended**. For simulation purposes, this interaction is automated— the Airflow script also makes calls to the FastAPI `/feedback` endpoint, mimicking user input. The simulated feedback is then stored in a **PostgreSQL database**, ensuring we can track and analyze user preferences over time.

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

[Airflow script to extract feedback data for retraining](./../model_monitoring/dags/pipeline_extract_prod_data.py)

The **user feedback data stored in PostgreSQL** is extracted on a weekly basis using a separate Airflow script. This script processes the data by splitting it into **training, validation, and test sets**, and converts it into the appropriate format required for retraining the model.

To store the processed data, the script uses the rclone tool (invoked via Python subprocesses) to upload the new datasets to the **object store**. This newly collected data is then appended to the existing training dataset.

In recommendation systems, it's common to adopt a **sliding window approach** for training data. As we accumulate more feedback, older data is gradually phased out to keep the dataset manageable in size and relevant. This is especially important in our context—since user preferences tend to evolve, recent behavior (e.g., from the past month) is generally more indicative of current interests than data that is several months or a year old.



# Data Preprocessing

## General overview

The data preprocessing was one of they key procedures for our project, and one of the tasks that took the longest toll on our time. The main reason is that we had to take the Million Playlist Dataset (MPD) and Echo-Nest Taste Profile dataset (ENTP) and merge them together, simulating what playlists (MPD) each user (ENTP) likes.

Additionally, the datasets were quite large. MPD itself is about 30GB and ENTP about 3GB. MPD has about 1 million playlists with about 2 million different songs. ENTP has 1 million users and about 380k songs. The song overlap between both datasets is 120k songs.

Midway through preprocessing we decided to resize the data, and we kept working only with 10% of users. The full-sized dataset was too large to work with, required to batch every single operation, frequently crashed and debugging was very difficult. Since the data was very large, we were able to afford downsizing and still have a good training process. More details given below.

---

---

## Part 1) Raw data and initial processing

- We load raw data both from ENTP and MPD. Nothing is done here to MPD data, apart from putting the raw slices in a directory.
- For ENTP data, raw data only has song_id’s, but no song names or artist names. This info is supposed to be fetched from the Million Song Dataset (different from MPD): [http://millionsongdataset.com/](http://millionsongdataset.com/). We grab that information and create a new ENTP dataset with that information included.

---

---

## Part 2) Processing MPD and ENTP and create unique song_ids shared between the two datasets

- As mentioned there are about 120k songs shared among MPD and ENTP. In this step we generate unique song id’s that are shared between MPD and ENTP.

---

---

## Part 3) Compute the overlap between users and playlists based on songs: use of sparse matrices (scipy.sparse)

- Goal: we need to assess for each user and playlist, how many songs of the playlist exist in the user song collection. We are looking for a shape like the one in the table below:

| user_id | playlist_id | song matches |
| --- | --- | --- |
| 42 | 123 | 10 |
| 42 | 423 | 35 |
| 107 | 123 | 0 |
| 107 | 423 | 24 |
- Initially we started to compute this iteratively, but the estimated time to complete the calculation was about a week, so we decided to use sparse matrices instead.
- For the sparse matrices, we create first a user_song matrix and also a playlist_song matrix. The user_song matrix looks something like the table below, where 1 indicates the user has that song in its library. The same is done for playlists.

| user_id/song_id | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 1 | 1 | 1 | 0 |
| 2 | 0 | 0 | 0 | 0 | 0 |
| 3 | 1 | 0 | 1 | 0 | 1 |
| 4 | 1 | 0 | 1 | 1 | 1 |
| 5 | 1 | 1 | 0 | 1 | 0 |
- Then, we take the dot product between both matrices. Both matrices are huge, therefore we batch the multiplication per slices. The final result looks like the table below:

| user_id/playlist_id | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 12 | 1 | 5 | 25 | 51 |
| 2 | 0 | 0 | 1 | 2 | 3 |
| 3 | 12 | 2 | 5 | 25 | 51 |
| 4 | 62 | 4 | 5 | 42 | 24 |
| 5 | 131 | 3 | 5 | 24 | 8 |
- Finally we grab this huge matrix and reshape it in a more friendly shape like the table below:

| user_id | playlist_id | song_matches |
| --- | --- | --- |
| 1 | 1 | 12 |
| 1 | 2 | 1 |
| 1 | 3 | 5 |
| 2 | 1 | 0 |
| 2 | 2 | 1 |

## Part 3.5) Getting the song count per user and song count per playlist (will be required for Part 4):

- Here we just compute the size of the song library of each user, and the amount of songs in each playlist and store it in a file. We will need this for Part 4.

---

---

## Part 4) Calculate scores per user-playlist pairs

- For each user-playlist pair we need a score of how much the user ‘likes’ the given playlist. This scores will then be used to create the true labels for training. Basically each user will have ‘liked’ playlists, that they would like if they are given as a recommendation.
- Our scoring function is of course based on song similarity between the user song library and the playlist songs list. However, we added some more complexity to it, captured in an ‘exploration coefficient’ for each user. This coefficient goes from 0 to 1 and measures ‘how explorative’ each user is. A user with exploration_coefficient = 0 wants playlists that are perfectly aligned with their current song library, a user with exploration_coefficient=1 values only playlists that have no overlap with their current song library.
- Additionally, the scores are normalized both by user song library size and by playlist song size.

The score between a user \( i \) and a playlist \( j \) is computed as:

$$
\text{Score}_{ij} = \frac{(1 - e) \cdot \log(1 + m) + e}{\sqrt{s} \times \sqrt{p}}
$$

Where:
m = number of song matches between user i and playlist j
e = exploration coefficient for user i (random between 0 and 1, skewed towards lower values using a Beta distribution)
s = total number of unique songs in user i's library
p = total number of unique songs in playlist j

Higher scores indicate playlists better aligned to the user's taste.
The e term adds randomness and encourages exploration.
We normalize by s and p to avoid favoring users or playlists with large libraries.

- For the exploration coefficient, we sampled values from a beta distribution. The graph below shows it’s distribution. The assumption is the following: most users want to explore a little bit, and don’t want playlists 100% aligned with their current song library. Also, a few users are very explorative, but the distribution falls quickly as the exploration coefficient increases.


![Exploration Coefficient Distribution](data_preprocessing/images_log/image.png)


- We are also including some statistics about the exploration coefficient: s

```
📊

Exploration Coefficient Statistics:
Min: 0.0000
P1: 0.0096
P25: 0.1287
Median: 0.2496
Mean: 0.2827
P75: 0.4064
P99: 0.7835
Max: 0.9900

```

- Finally, we compute the scores for all user-playlist pairs. The min score sits between 0.0013 and 0.0016, the max score is around 0.49, and the medians core is around 0.125.

---
---

## Part 5) Rank playlists per user and tag ‘liked’ playlists

- For each user we will take the top 1.5% playlists by score and mark them as ‘liked’ by the user.
- Some statistics for the amount of liked playlists per user below (this is for just one slice of 10,000 users). We can see that the median is about 200 songs liked per user, which is reasonable if we think about an average person and the potential set of playlists they might like.

```
📊

Statistics of liked playlists per user, for a slice of users:
Total users: 9,984
Total rows: 223,192,860
Positive (liked) playlists: 3,352,842 (1.50%)
Negative (non-liked) playlists: 219,840,018 (98.50%)
Average playlists liked per user: 335.82
Median playlists liked per user: 199.00
Min playlists liked per user: 1.00
Max playlists liked per user: 4,339.00

```

![Distribution of Liked Playlists per User](data_preprocessing/images_log/image_1.png)

---

---

## Part 6) Start preparing the splits of data: new_users (cold users), and then train, validation, and test splits. Also assign playlist ownership among users.

- This section has 4 sub-steps:
1. Downscale the dataset to 10%. Every operation was taking too long and the dataset was too massive. We sample here 10% of users and continue only with them.
2. Split the data
    1. First, we will separate from the data 5% of users, with all of their playlist information. This will be kept aside as cold users for post-deployment tasks.
    2. Next, we split the liked playlists per user into training, validation and test sets. This is to resemble the ‘past and future’ nature of recommender systems. We will train the model on user-playlist interactions we have seen so far, but for evaluation, we will ask the model to rank at the top a playlist that we know the user should like, but that the model hasn’t seen as a positive for that user yet. We do a 70%-20%-10% split between train-val-test sets, among the positive playlists (”liked” playlists).

---

---

## Part 7) Assign Playlist Ownership

1. Assign playlist ownership
    1. We will assign playlist ownership in principle to the user that has the highest score for that playlist. However, we will also normalize by the user’s song library size, to avoid assigning a playlist to a user only because their song library is huge (’generalist user’).
    2. We will calculate then a ‘priority score’ as below:
$$
\text{priority} = \frac{\text{score}}{\sqrt{\text{user's song count}}}
$$
    4. Validation positives are excluded from ownership assignment for a given user.
    5. We also impose a maximum amount of playlists owned by a single user to 100 playlists.
    6. The algorithm iterates over playlists and looks into the highest scoring users for that playlist. It guarantees that every single playlist will be owned by one and only one user.
    7. Let’s explore some statistics about playlist ownership below. We see that on average users own 4.6 playlists, with a median of 2 playlists. We also explore some ‘ inequality’ metrics: 1% of users own 15% of playlists, and 25% own 72% of playlists. We believe this is quite realistic, as in music streaming services not all users are heavy producers of their own playlists, and some users are ‘power users’  that create many playlists. Of course these are mere assumptions.
    
    ```
    📊
    
    Playlist Ownership Stats (Global):
    
    - Users with ownership: 39,653
    - Min owned : 1
    - Max owned : 100
    - Average owned : 4.57
    - Median owned : 2.0
    - 25th percentile : 1.0
    - 75th percentile : 4.0
    ```
    
    ![Histogram of Playlist Ownership per User](data_preprocessing/images_log/image_2.png)
    
    ```
    📊
    
    Playlist Ownership Inequality Stats:
     - Top 1% own 15.34% of playlists
     - Top 5% own 37.29% of playlists
     - Top 10% own 50.83% of playlists
     - Top 25% own 71.68% of playlists
     - Gini coefficient       : 0.6033
    
    ```
    
    ![Lorenz Curve of Playlist Ownership (straight line is perfect equality)](data_preprocessing/images_log/image_3.png)
    
    ---
    
    ---
    
    ## Part 8) Generate triplets training dataset!
    
    - Bayesian Personalized Ranking (BPR) models are typically trained using triplets that express a **relative user preference**.
    - Each triplet has the form: `(user_id, positive_playlist_id, negative_playlist_id)`.
    - We use the training split of positive playlists, and for each of them, we generate **4 negatives**.
    - Negative playlists are drawn from a precomputed pool of candidates:
        - **Top 20% (by score)** are considered **hard negatives**—they were close to being positives and help the model learn to distinguish fine-grained preferences.
        - The remaining 80% are **easy negatives**.
    - For each positive, we sample:
        - Half the negatives from the **hard pool**, and
        - The other half from the **easy pool**.
    - We avoid including playlists that were **owned or already positively labeled**, ensuring clean contrastive signals.
    - The script also creates a **toy dataset** for 5% of users to support fast debugging and experimentation.
    - Finally we shuffle and save the file in torch format.
    - Below we can see some statistics about the triplets. On average, each user has 942 triplets, with a median of 576.
    
    ```
    📊
    
    Stats for one slice of triplets:
    
    - Users in slice : 945
    - Total triplets : 890,860
    - Mean : 942.71
    - Min : 4
    - Max : 10532
    - P1 : 4.00
    - P25 : 152.00
    - P50 (Median) : 576.00
    - P75 : 1248.00
    - P99 : 6686.72
    ```
    
    ![Triplets per User (for one slice of data)](data_preprocessing/images_log/image_4.png)
    

---

---

## Part 9) Generate validation batches

- We will do our BPR model evaluation during training in a customized way (that is standard for recommendation models).
- We will pack 1 unseen positive for a given user with 50 negative playlists, and we will measure how often and how high does the model rank the positive playlist.
- We will use two types of metrics:

    - **Hit@K:**
        
        Measures the percentage of times the positive playlist appears in the top K ranked results. It answers the question: *"Did we get the right answer in the top K?"*
        
    - **MRR (Mean Reciprocal Rank)**
        
        Measures the average of the reciprocal ranks of the true positive in the ranked list. A higher MRR means the model is ranking positives closer to the top.
        
$$
\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}
$$
        

- Therefore, we need to prepare validation batches for each user. 1 unseen positive packed with 50 negatives. Negatives can’t be playlists that were positives in training.
- For a given user, we will have both unseen and seen negatives (seen negatives = negatives that were negatives for that user in training).
- The validation positives are unseen for a given user, but they might have been a positive for a different user during training.


## Continuous X (Jishnu)

- We will implement a fully automated CI/CD pipeline using Github Actions and integrate with the infrastructure hosted on ChameleonCloud
- All the cloud native principles such as containerization, microservices, immutable infrastructure and version controlled automated deployments will be used.
- Containerization
    - All components of the system (model training, inference APIs and monitoring tools) will be packaged as docker containers. This will ensure reproducibility, modularity and easy deployment across environments.
- Orchestration
    - We will use Kubernetes on Chameleon to manage our containerized services.
    - Kubernetes will handle:
        - Orchestration of training and serving containers
        - Health checks and restart policies
        - autoscaling of servers based on resource usage or request load
        - Rolling updates and rollbacks for safe deployments
- The pipeline will be triggered by:
    - Code pushes to the main branch
    - Scheduled training
    - Performance degradation alerts from monitoring
- Pipeline steps:
    - Have webhooks for unit testing before pushing
    - Retrain both the user pairing and playlist ranking models using Ray
    - Run offline evaluation and log metrics to MLflow
    - Build Docker containers for each service
    - Package services into Docker containers and deploy to staging, canary, and production environments.
- Deployment environments
    - Staging
        - All services will be deployed here for our load and integration testing
    - Canary
        - New models will be rolled out to a subset of users for online evaluation
        - If the canary evaluation passes our engagement and accuracy metrics
    - Production
        - Final stable version of the system that are available to all the users.

- Infrastructure as code
    - The infrastructure will be provisioned using Ansible and Helm
    - Infrastructure will follow immutable infrastructure principles—changes are only made by updating configuration and redeploying

