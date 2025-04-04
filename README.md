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
| Agustin Leon | Model training, experiment tracking (Units 4 & 5)  |  |
| Akhil Manoj | Data pipeline (unit 8) |  |
| Anup Raj Niroula | Model serving and monitoring (Units 6 & 7) |  |
| Jishnu Warrier | Continuous pipeline (unit 3) |  |

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

| Unit | Topic | Difficulty point |
| --- | --- | --- |
| 1 | ML Ops Intro Unit | Using multiple models (we’ll be using two models) |
| 5 | Model training infrastructure and platform | Ray Train |
| 6 | Model serving and monitoring | Develop multiple options for serving |
| 7 | Evaluation and monitoring  | Monitor for model degradation |
| 8 | Data Pipeline | Interactive Data Dashboard |

# Detailed design plan

## Model Training & Infrastructure (Agustin)

### Model 1: User Pairing Model

- **Strategy:**
    - The user pairing model will use **Embedding-Based K-Nearest Neighbors (KNN)** to find similar users based on their music taste.
    - Each user is represented as a vector by aggregating the embeddings of the songs they like. For this, we will use **precomputed song embeddings** (e.g., via Word2Vec, TF-IDF, or matrix factorization), and generate user embeddings by taking the mean of the vectors corresponding to their liked songs.
    - To find similar users, we compute the **cosine similarity** between the target user’s embedding and all other users in the dataset, and select the top-K most similar ones.
    - This approach is unsupervised and does not require training, but inferences might take long. This is not an issue because we are planning on running the user pairing model much less frequently than the playlist ranker model. We will compute the pairings, and then we can run the playlist ranker model multiple times based on the same pairings. The exact frequency of how often the pairings must update will be determined later based on the data pipeline and user interaction (but probably something like once a week or once a month).
    - This model naturally supports **discovery** through social similarity: if UserA and UserB both like indie rock, but User B also likes niche electronic playlists, UserA may get exposed to those too — even if they haven’t listened to electronic music yet.

### Model 2: Playlist Ranker Model

- S**trategy:**
    - The playlist ranker will use Bayesian Personalized Ranking (BPR), which we’ve learned is a latent factor model.
    - BPR basically learns embeddings for users and items (e.g., playlists) by optimizing a ranking objective using implicit feedback. It uses gradient-based optimization, with optimizers such as SGD or Adam.
    - For each user, BPR is trained to distinguish between playlists they like and those they do not by optimizing a pairwise ranking loss. This approach is well-suited for our use case, where users interact with playlists by liking them, without providing explicit ratings ( we don’t know how much they liked it, just if they like it or not).
    - Familiarity/serendipity trade-off is key for recommender models. We will incorporate features such as overlap with user-liked songs, novelty (unheard songs), and playlist diversity into the scoring function. By optimizing playlist rankings in a user-specific manner, BPR helps the system recommend organic playlists that match user taste while also encouraging exploration of new music curated by similar users.

### General training comments for both models

- Inf**rastructure:**
    - BPR should not require excessive resources. We should have plenty with one A100 GPU. KNN should also run properly with the A100 GPU.
- **Experiment Tracking:**
    - ML Flow hosted on Chameleon. We will log hyperparameters, metrics, and artifacts
- **Unit 4 specific comments:**
    - We will re-train the models, with new labeled data.
- **Unit 5 specific comments:**
    - We will track experiments using ML Flow and we will track features such as fault tolerance, and checkpointing using Ray Train.
    - We will also schedule hyperparameter tuning jobs using Ray Tune.

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

## Data pipeline (Akhil)

- Offline Data - Process data from the Million Playlist Dataset and the EchoNest Taste Profile dataset and store them in separate tables in a MongoDB database. The dataset is in JSON format, so storing in MongoDB is an appropriate choice.
- Persistent Storage - Provision a persistent storage on Chameleon using MLFlow for storing model, model artifacts and test artifacts (part of model serving)
- Data Pipelines - We will have an ETL pipeline to ingest new data from our production service based on users interacting with the songs in our recommended playlist (and other songs in general). This new data has to be parsed and processed based on the user’s activity of the platform and added to the data repository.
- Online data - Will be simulated by us interacting with songs and/or by a script which mimics user behavior. We would interact with the songs in the playlist (and other songs in general) which would give us insight on new playlist recommendations for the user. The ETL pipeline (from above) will be used to process the users interaction with the songs and add them into would data repository. Model training would be done on this new data.
- Interactive data dashboard(extra difficulty) - Interactive dashboard to get insights about the users interacting with the songs in the playlist, listening time on organic playlists opposed to other playlists, most popular recommended playlists(for new users) and other similar metrics.

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
