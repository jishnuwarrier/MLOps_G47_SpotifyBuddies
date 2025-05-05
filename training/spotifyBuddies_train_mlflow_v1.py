# === 1. Setup ===
import os, random, torch, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pickle
import glob
import polars as pl

# === 1.1 MLFlow Setup ===
import mlflow
import mlflow.pytorch

print("1/5: Starting job")

# === 2. Config ===
BASE_DIR = '/mnt/block'
BASE_DIR_OBJ = '/mnt/data'
DATASET_NAME = 'easy_toy'
OUTPUT_DATASET_NAME = 'easy_toy'
TRIPLET_DATA_PATH = BASE_DIR
METADATA_DIR = BASE_DIR_OBJ
# CHECKPOINT_DIR = os.path.join(BASE_DIR, f'training_output/{OUTPUT_DATASET_NAME}/checkpoints')
# TENSORBOARD_DIR = os.path.join(BASE_DIR, f'training_output/{OUTPUT_DATASET_NAME}/tensorboard')
# os.makedirs(TENSORBOARD_DIR, exist_ok=True)
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TRIPLETS_DATASET_NAME = 'all_triplets.pt'
VAL_POSITIVES_NAME = 'val_positives.pkl'



EMBEDDING_DIM = 128
BATCH_SIZE = 16384
GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 0.005
EPOCHS = 5
EARLY_STOPPING_PATIENCE = 5
RESUME_FROM_CHECKPOINT = False
# CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pt')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = torch.cuda.is_available()


# === 4. Load Triplets ===
print("2/5: Loading dataset...")
triplets = torch.load(os.path.join(BASE_DIR_OBJ, TRIPLETS_DATASET_NAME))
print("3/5: Dataset loaded.")
NUM_USERS = triplets[:, 0].max().item() + 1
NUM_PLAYLISTS = triplets[:, [1, 2]].max().item() + 1

# === 5. Dataset + DataLoader ===
class BPRTripletDataset(Dataset):
    def __init__(self, triplet_tensor):
        self.triplet_tensor = triplet_tensor

    def __len__(self):
        return self.triplet_tensor.shape[0]

    def __getitem__(self, idx):
        return tuple(self.triplet_tensor[idx])

train_loader = DataLoader(
    BPRTripletDataset(triplets),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

# === 6. BPR Model ===
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

# === 7. Load Validation Positives ===
with open(os.path.join(METADATA_DIR, VAL_POSITIVES_NAME), 'rb') as f:
    val_positives_full = pickle.load(f)

VAL_EVAL_SAMPLE_RATIO = 0.2
sampled_eval_users = random.sample(list(val_positives_full.keys()), int(len(val_positives_full) * VAL_EVAL_SAMPLE_RATIO))
val_positives = {uid: val_positives_full[uid] for uid in sampled_eval_users}

# === 8. Evaluation Function ===
def evaluate_full_ranking(model, val_positives, k_list=[1, 5, 10]):
    model.eval()
    with torch.no_grad():
        all_playlist_ids = torch.arange(model.playlist_embeddings.num_embeddings, device=device)
        playlist_emb = model.playlist_embeddings(all_playlist_ids)

        hit_counts = {k: 0 for k in k_list}
        mrr_total = 0
        total = 0

        for user_id, positives in tqdm(val_positives.items(), desc="🔎 Full Ranking Eval"):
            user_tensor = torch.tensor([user_id], device=device)
            user_emb = model.user_embeddings(user_tensor)
            scores = torch.matmul(user_emb, playlist_emb.T).squeeze(0)

            for seen_pid in positives:
                scores[seen_pid] = -1e9

            top_preds = torch.topk(scores, max(k_list)).indices.tolist()
            rank = [i for i, pid in enumerate(top_preds) if pid in positives]
            if rank:
                mrr_total += 1 / (rank[0] + 1)
            for k in k_list:
                if any(pid in positives for pid in top_preds[:k]):
                    hit_counts[k] += 1
            total += 1

        hit_rates = {k: hit_counts[k] / total for k in k_list}
        mrr = mrr_total / total
        return hit_rates, mrr

# === 9. Initialize Model + Resume ===
model = BPRModel(NUM_USERS, NUM_PLAYLISTS, EMBEDDING_DIM).to(device)
optimizer = torch.optim.SparseAdam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler(enabled=use_amp)
# writer = SummaryWriter(log_dir=TENSORBOARD_DIR)

start_epoch = 1
BEST_VAL_MRR = -1
patience_counter = 0

# if RESUME_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
#     print(f"🔁 Resuming from checkpoint: {CHECKPOINT_PATH}")
#     checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     BEST_VAL_MRR = checkpoint['best_val_mrr']
#     start_epoch = checkpoint['epoch'] + 1
#     print(f"✅ Resumed at epoch {start_epoch} with BEST_VAL_MRR = {BEST_VAL_MRR:.4f}")
# else:
print("Starting fresh training run.")

# === 10. MLFlow Start + Log Params ===
mlflow.set_experiment(f"SpotifyBuddies_experiment1")
try:
    mlflow.end_run()
except:
    pass
finally:
    mlflow.start_run(log_system_metrics=True)

mlflow.log_params({
    "embedding_dim": EMBEDDING_DIM,
    "batch_size": BATCH_SIZE,
    "grad_accum_steps": GRAD_ACCUM_STEPS,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "dataset": DATASET_NAME,
    "num_users": NUM_USERS,
    "num_playlists": NUM_PLAYLISTS
})

# === 11. Training Loop ===
for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    total_loss = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (user_ids, pos_pids, neg_pids) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        user_ids = user_ids.to(device, non_blocking=True)
        pos_pids = pos_pids.to(device, non_blocking=True)
        neg_pids = neg_pids.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            pos_scores, neg_scores = model(user_ids, pos_pids, neg_pids)
            loss = bpr_loss(pos_scores, neg_scores) / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * GRAD_ACCUM_STEPS
        del user_ids, pos_pids, neg_pids, pos_scores, neg_scores, loss

    avg_train_loss = total_loss / len(train_loader)
    val_hit_rates, val_mrr = evaluate_full_ranking(model, val_positives)

    print(f"📊 Epoch {epoch} | Loss: {avg_train_loss:.4f} | Val MRR: {val_mrr:.4f} | Hit@10: {val_hit_rates[10]:.4f}")

    # writer.add_scalar('Loss/train', avg_train_loss, epoch)
    # writer.add_scalar('Val/MRR', val_mrr, epoch)
    # for k in val_hit_rates:
    #     writer.add_scalar(f'Val/Hit@{k}', val_hit_rates[k], epoch)

    mlflow.log_metrics({
        "train_loss": avg_train_loss,
        "val_mrr": val_mrr,
        **{f"hit@{k}": val_hit_rates[k] for k in val_hit_rates}
    }, step=epoch)

    # Checkpointing
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_mrr': BEST_VAL_MRR
    }

    if val_mrr > BEST_VAL_MRR:
        BEST_VAL_MRR = val_mrr
        patience_counter = 0
        # torch.save(checkpoint_data, os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}_mrr_{val_mrr:.4f}.pt'))
        # torch.save(checkpoint_data, CHECKPOINT_PATH)
        mlflow.pytorch.log_model(model, artifact_path="bpr_model")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"⏹️ Early stopping at epoch {epoch}.")
            mlflow.log_metric("early_stopping_epoch", epoch)
            break

# === 12. Cleanup ===
# writer.close()
mlflow.end_run()
print("✅ Training complete.")
