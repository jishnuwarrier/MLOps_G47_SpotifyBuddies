## Training beautiful stuff and sending it to MLFlow

[reference](https://edstem.org/us/courses/74594/discussion/6556846)

1. You need 2 instances - 1 on chi:tacc (has GPU) and other on kvm:tacc (runs mlflow, minio, postgres etc)

### On chi:tacc instance
1. Once your lease begins, follow gpu_init.ipynb file on Chameleon to get a GPU instance
2. Make sure it can access object store (follow data-pipeline/README)
3. Run the jupyter-mlflow image command. Make sure MLFLOW_TRACKING_URI is the ip of kvm:tacc instance
4. Clone the repo
```
git clone https://github.com/AguLeon/MLOps_G47_SpotifyBuddies
```
5. 
```
docker run  -d --rm  -p 8888:8888 \
    --gpus all \
    --shm-size 16G \
    -v ~/MLOps_G47_SpotifyBuddies:/home/jovyan/work/ \
    --mount type=bind,source=/mnt/object,target=/mnt/data,readonly \
    -e MLFLOW_TRACKING_URI=http://129.114.27.215:8000/ \
    -e OBJECT_DATA_DIR=/mnt/object \
    --name jupyter \
    jupyter-mlflow
```

### On kvm:tacc instance
1. The kvm:tacc instance should have the block storage mounted (this is done from the UI. already done for our instance)
2. Run docker-compose-block.yaml file which has mlflow, minio, postgres etc
