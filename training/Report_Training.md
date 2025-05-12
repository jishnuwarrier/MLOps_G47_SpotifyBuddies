# Training README

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

![Distribution of Neighbor Count per User](images_log/image.png)

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

![Distribution of User Popularity](images_log/image_1.png)

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

![Distribution of Playlist Access via Neighbors](images_log/image_2.png)

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

![MRR across 5 epochs](images_log/image_3.png)

- Hit@5 is a very important metric because our model serving service will recommend 5 playlists to users. We can see that in 72% of cases a positive playlist existing in a bag with 50 negative playlists, will be ranked inside the top 5 playlists.

![Hit@5 across 5 epochs](images_log/image_4.png)

### Ray Tune results

- With respect to Ray Tune, we are including below a comparison of the experiments that Ray Tune handled. Since we ran Ray Tune just one time on the full dataset, we were able to explore only variations on the learning rate.

![Comparison of learning rates for Ray Tune experiments](images_log/image_5.png)

- We can see that Ray Tune identified some space for improvement, but it was very minor. Probably we need to experiment further in order to discover more significant improvements. Basically we tried too little of variations of learning rates, that allowed to increase MRR from 0.47889 to 0.47912, which is basically negligible.

![Comparison of metrics (MRR and Hit@5) for Ray Tune experiments](images_log/image_6.png)

- Probably it would have been more interesting to explore another hyperparameter, such as embedding dimensions of the BPR Model, but still Ray Tune proved to be a valuable tool at exploring hyperspace optimization in a consistent way.