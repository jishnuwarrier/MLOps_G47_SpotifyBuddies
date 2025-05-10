print("Starting training job! Let's go")

# === 1. GENERAL PARAMETERS ===
SEED = 42
USE_RAY_TUNE = True
USE_MLFLOW = True
RUN_ON_CHAMELEON = True
TRAIN_FULL_DATASET = False   # Select False for training on a toy dataset for testing and quick debugging

# === 2. IMPORT DEPENDENCIES ===
import os
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import random
import pickle
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import re
import shutil
from ray import tune
from ray.train import report


# === 3. DIRECTORIES AND SAVING OPTIONS ===
print("-Imports done. Setting directories now.")

TRAIN_TOY_DATASET = not TRAIN_FULL_DATASET
TRIPLETS_DATASET_FILENAME = "training_triplets_toy.pt" if TRAIN_TOY_DATASET else "training_triplets_full.pt"
TRIPLETS_DATASET_NAME = "training_triplets_toy" if TRAIN_TOY_DATASET else "training_triplets_full"

if RUN_ON_CHAMELEON:
    BASE_DIR = '/mnt/object'
    LOCAL_BASE_DIR = '.'
    OUTPUT_DIR = os.path.join(LOCAL_BASE_DIR, f'training_output/{TRIPLETS_DATASET_NAME}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    CHECKPOINT_DIR = os.path.join(LOCAL_BASE_DIR, f'training_output/{TRIPLETS_DATASET_NAME}/checkpoints')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
else:
    BASE_DIR = '/content/gdrive/MyDrive/datasets/main_datasets'
    OUTPUT_DIR = os.path.join(BASE_DIR, f'training_output/{TRIPLETS_DATASET_NAME}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    CHECKPOINT_DIR = os.path.join(BASE_DIR, f'training_output/{TRIPLETS_DATASET_NAME}/checkpoints')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TRIPLET_DATASET_PATH = os.path.join(BASE_DIR, f'triplets_dataset/{TRIPLETS_DATASET_FILENAME}')

VAL_EVAL_BASE_DIR = os.path.join(BASE_DIR, 'val_eval_batches')
VAL_EVAL_SUBDIR = 'val_eval_batches_toy' if TRAIN_TOY_DATASET else ''
VAL_EVAL_BATCH_DIR = os.path.join(VAL_EVAL_BASE_DIR, VAL_EVAL_SUBDIR) if VAL_EVAL_SUBDIR else VAL_EVAL_BASE_DIR

MODEL_CONFIG_PATH = os.path.join(CHECKPOINT_DIR, 'model_config.json')
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'final_model.pt')
FINAL_FULL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'final_full_checkpoint.pt')


# === 4. TRAINING PARAMETERS ===
print("-Directories done. Setting parameters.")
EMBEDDING_DIM = 128
BATCH_SIZE = 16384
GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 0.005
EPOCHS = 3
EARLY_STOPPING_PATIENCE = 5
RESUME_FROM_CHECKPOINT = False

# === 5. MLFLOW PARAMETERS ===
MLFLOW_EXPERIMENT_NAME = 'SpotifyBuddies_experiment4_RayTune'
MLFLOW_TAGS = {
    "platform": "chameleon_cloud",
    "mode": "toy",
    "run_type": "baseline"
}

# === 6. RAY TUNE PARAMETERS ===
try:
    from ray import tune
    RAY_TUNE_AVAILABLE = True
except ImportError:
    RAY_TUNE_AVAILABLE = False
    print("Ray Tune not available!")

RAY_NUM_SAMPLES = 6
RAY_SEARCH_SPACE = {
    "embedding_dim": tune.choice([64, 128]) if RAY_TUNE_AVAILABLE else EMBEDDING_DIM,
    "learning_rate": tune.loguniform(1e-4, 1e-2) if RAY_TUNE_AVAILABLE else LEARNING_RATE,
    "batch_size": tune.choice([8192, 16384]) if RAY_TUNE_AVAILABLE else BATCH_SIZE,
    "val_batch_dir": VAL_EVAL_BATCH_DIR,
    "triplet_path": TRIPLET_DATASET_PATH
}


# === 7. LOAD DATASET ===
print("-Loading data.")
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = torch.cuda.is_available()

triplets = torch.load(TRIPLET_DATASET_PATH)
NUM_USERS = triplets[:, 0].max().item() + 1
NUM_PLAYLISTS = triplets[:, [1, 2]].max().item() + 1

class BPRTripletDataset(Dataset):
    def __init__(self, triplet_tensor):
        self.triplet_tensor = triplet_tensor

    def __len__(self):
        return self.triplet_tensor.shape[0]

    def __getitem__(self, idx):
        return tuple(self.triplet_tensor[idx])
    



# === 8. MODEL ===
print("-Creating model.")
class BPRModel(nn.Module):
    def __init__(self, num_users, num_playlists, embedding_dim):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim, sparse=True)
        self.playlist_embeddings = nn.Embedding(num_playlists, embedding_dim, sparse=True)
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.playlist_embeddings.weight)

    def forward(self, user_ids, pos_pids, neg_pids):
        u = self.user_embeddings(user_ids)
        i = self.playlist_embeddings(pos_pids)
        j = self.playlist_embeddings(neg_pids)
        return (u * i).sum(1), (u * j).sum(1)

def bpr_loss(pos_scores, neg_scores):
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

# === 9. EVALUATION ===
def load_val_eval_batches(base_dir, slice_ids=None):
    batches = []

    if isinstance(slice_ids, int):
        slice_ids = [slice_ids]

    if slice_ids is not None:
        for slice_id in slice_ids:
            path = os.path.join(base_dir, f"val_eval_batch_{slice_id}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    batches.extend(pickle.load(f))
            else:
                print(f"⚠️ Slice not found: {path}")
    else:
        for fname in os.listdir(base_dir):
            if fname.endswith(".pkl"):
                with open(os.path.join(base_dir, fname), "rb") as f:
                    batches.extend(pickle.load(f))

    return batches


def evaluate_ranking(model, base_dir, slice_id=None, k_list=[1, 5, 10]):
    model.eval()
    hit_counts = {k: 0 for k in k_list}
    mrr_total = 0
    total_samples = 0

    val_eval_batches = load_val_eval_batches(base_dir, slice_id if slice_id is not None else None)

    with torch.no_grad():
        for user_id, pos_pid, neg_pids in tqdm(val_eval_batches, desc="Evaluating"):
            user_tensor = torch.tensor([user_id], device=device)
            full_list = [pos_pid] + neg_pids
            candidate_tensor = torch.tensor(full_list, device=device)
            user_expand = user_tensor.expand(len(full_list))

            scores, _ = model(user_expand, candidate_tensor, candidate_tensor)
            _, ranking_indices = torch.sort(scores, descending=True)

            rank_pos = (ranking_indices == 0).nonzero(as_tuple=True)[0].item()

            mrr_total += 1 / (rank_pos + 1)
            for k in k_list:
                if rank_pos < k:
                    hit_counts[k] += 1
            total_samples += 1

    if total_samples == 0:
        print("Warning: No validation samples found.")
        return {k: 0.0 for k in k_list}, 0.0

    hit_rates = {k: hit_counts[k] / total_samples for k in k_list}
    mrr = mrr_total / total_samples
    return hit_rates, mrr



# Select which slices to run validation on

SELECTED_VAL_SLICES = [0] #Select slices among 0, 1, 2, 3, 4. Can select all as well [0, 1, 2, 3, 4]. Each slice takes about 8 minutes to evaluate


del triplets
# === 10. TRAINING FUNCTION ===
def train_fn(config):
    if USE_RAY_TUNE and RAY_TUNE_AVAILABLE:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    model = BPRModel(NUM_USERS, NUM_PLAYLISTS, config["embedding_dim"]).to(device)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=config["learning_rate"])
    scaler = GradScaler(enabled=use_amp)
    triplets = torch.load(config["triplet_path"])
    train_loader = DataLoader(
        BPRTripletDataset(triplets),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    start_epoch = 1
    best_val_mrr = -1
    patience_counter = 0

    if USE_MLFLOW:
        mlflow.set_tracking_uri("http://129.114.27.215:8000")
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        mlflow.start_run()
        mlflow.log_params(config)
        mlflow.set_tags(MLFLOW_TAGS)

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (user_ids, pos_pids, neg_pids) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            user_ids = user_ids.to(device)
            pos_pids = pos_pids.to(device)
            neg_pids = neg_pids.to(device)

            with autocast(enabled=use_amp):
                pos_scores, neg_scores = model(user_ids, pos_pids, neg_pids)
                loss = bpr_loss(pos_scores, neg_scores) / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * GRAD_ACCUM_STEPS

        torch.cuda.empty_cache()
        avg_train_loss = total_loss / len(train_loader)
        val_hit_rates, val_mrr = evaluate_ranking(model, base_dir=config["val_batch_dir"], slice_id=SELECTED_VAL_SLICES)

        hit_str = " | ".join([f"Hit@{k}: {val_hit_rates[k]:.4f}" for k in sorted(val_hit_rates)])
        print(f"➡️➡️ Epoch {epoch} | Loss: {avg_train_loss:.4f} | Val MRR: {val_mrr:.4f} | {hit_str}")

        if USE_MLFLOW:
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_mrr": val_mrr,
                **{f"hit-{k}": val_hit_rates[k] for k in val_hit_rates}
            }, step=epoch)

        if RAY_TUNE_AVAILABLE and USE_RAY_TUNE:
            # tune.report(train_loss=avg_train_loss, val_mrr=val_mrr, **{f"hit@{k}": val_hit_rates[k] for k in val_hit_rates})
            report({"train_loss": avg_train_loss, "val_mrr": val_mrr, **{f"hit@{k}": val_hit_rates[k] for k in val_hit_rates}})


        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_mrr": best_val_mrr
            }, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}_mrr_{val_mrr:.4f}.pt"))
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                if USE_MLFLOW:
                    mlflow.log_metric("early_stopping_epoch", epoch)
                break

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_mrr": best_val_mrr
    }, FINAL_FULL_CHECKPOINT_PATH)


    if USE_MLFLOW:
        # Create a temp directory to hold the artifact (bc log_artifacs() is made to send folders (not files) to MLFlow)
        checkpoint_bundle_dir = os.path.join(OUTPUT_DIR, "checkpoint_bundle")
        os.makedirs(checkpoint_bundle_dir, exist_ok=True)

        # Copy the checkpoint file to this directory
        shutil.copy(FINAL_FULL_CHECKPOINT_PATH, checkpoint_bundle_dir)

        # Log the entire folder to MLflow (safer for large files)
        mlflow.log_artifacts(checkpoint_bundle_dir, artifact_path="checkpoints")

    with open(MODEL_CONFIG_PATH, 'w') as f:
        json.dump({
            "embedding_dim": config["embedding_dim"],
            "num_users": NUM_USERS,
            "num_playlists": NUM_PLAYLISTS,
            "seed": SEED
        }, f)



    if USE_MLFLOW:
        mlflow.pytorch.log_model(model, artifact_path="final_model")

        run_id = mlflow.active_run().info.run_id
        with open("last_run_id.txt", "w") as f:
            f.write(run_id)
        mlflow.log_artifact("last_run_id.txt")

        # Log model architecture/config file
        if os.path.exists(MODEL_CONFIG_PATH):
            mlflow.log_artifact(MODEL_CONFIG_PATH)

        # Log full training config
        with open("training_config.json", "w") as f:
            json.dump(config, f, indent=2)
        mlflow.log_artifact("training_config.json")

        mlflow.end_run()



# === 11. MAIN EXECUTION ===
print("-And starting to train!")
default_config = {
    "embedding_dim": EMBEDDING_DIM,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "val_batch_dir": VAL_EVAL_BATCH_DIR,
    "triplet_path": TRIPLET_DATASET_PATH
}

if USE_RAY_TUNE and RAY_TUNE_AVAILABLE:
    from ray import tune

    train_with_resources = tune.with_resources(
        train_fn,
        resources={"cpu": 4, "gpu": 0.5}  # We should get two workers sharing the GPU
    )

    # tune.run(
    #     train_fn,
    #     config=RAY_SEARCH_SPACE,
    #     metric="val_mrr",
    #     mode="max",
    #     num_samples=RAY_NUM_SAMPLES,
    #     name="bpr_hpo",
    #     resources_per_trial={"cpu": 4, "gpu": 0.5}
    # )

    tune.run(
        train_with_resources,
        config=RAY_SEARCH_SPACE,
        metric="val_mrr",
        mode="max",
        num_samples=RAY_NUM_SAMPLES,
        name="bpr_hpo"
    )

else:
    train_fn(default_config)
