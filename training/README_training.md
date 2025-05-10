
# SpotifyBuddies Training Setup Guide

This guide walks you through setting up the full training environment using **Docker**, **Ray**, **MLflow**, and a **Makefile** for automation.



---

## 0. Start an instance


Once your lease begins, follow gpu_init.ipynb file on training folder
Make sure it can access object store (follow data-pipeline/README)

Use SSH to connect to the instance (connects to node1-cloud-project47 instance)

```
ssh -i ~/.ssh/id_rsa_chameleon cc@129.114.108.116
```

Use the correct floating IP

---

## 1. Clone the Repository

```bash
git clone https://github.com/AguLeon/MLOps_G47_SpotifyBuddies.git
cd MLOps_G47_SpotifyBuddies
```

---

## 2. Start Ray and MLflow Containers

Use the provided Docker Compose file to spin up the Ray and MLflow environment:

```bash
docker compose -f docker-compose.mlflow-ray.yaml up -d
```

This launches:
- `ray-head`: the Ray cluster head node.
- `mlflow-client`: (optional) a Jupyter container with MLflow support.

Both containers mount your code and share the GPU if available.

---

## 3. Create and Activate the Virtual Environment

You can do this manually:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Or run:

```bash
make venv
```

This will create the `venv/` directory and install all required packages.

---

## 4. Train the Model

### Train on local machine:

```bash
make train
```


## 5. Additional Make Targets

### Clean all build files and virtual environment:

```bash
make clean
```

### Run code linter:

```bash
make lint
```

---

## 6. Monitor MLflow

If you are tracking experiments remotely, open your browser and go to:

```
http://129.114.27.215:8000
```

Artifacts logged include:
- Model checkpoints
- Metrics per epoch
- Model configs
- Serialized models for inference

---

## 7. Notes

- The Docker setup binds `/mnt/object` as block storage in your containers.
- Make sure GPU support is enabled if available (or comment out `--gpus all` in Docker).
- If running on CPU only, Ray Tune will still function but slower.

---
