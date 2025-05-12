import pickle
from collections import Counter
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn

from config import settings
from utils import Singleton


# TODO => Remove after stable model is created
class SimpleRecommender(nn.Module):
    def __init__(self):
        super(SimpleRecommender, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


mainstream_user_id = settings.MAINSTREAM_USER_ID
# NOTE -> In Real Actual system, we would be fetching the
# User to Playlist from a datasource/database (Spotify uses vector database)
# This is the output from User Pairing which is then mapped to
# the actual playlists that belong to neighbours
if settings.USE_MODEL:
    with open(settings.USER_PLAYLIST_PATH, "rb") as f:
        user_to_playlists = pickle.load(f)
        print("User-Playlist-Loaded")

    playlist_counter = Counter()
    for playlists in user_to_playlists.values():
        playlist_counter.update(playlists)

    # Step 2: Score each user by the *total popularity* of their playlists
    user_scores = {
        user: sum(playlist_counter[p] for p in playlists)
        for user, playlists in user_to_playlists.items()
    }

    # Step 3: Find user with highest popularity score
    mainstream_user_id = max(user_scores.items(), key=lambda x: x[1])[0]
    print(f"Most mainstream user: {mainstream_user_id}")


@dataclass
class Recommender(metaclass=Singleton):
    _model: nn.Module | None = None
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, path: str) -> None:
        """
        Initialize the model by loading the
        trained model from specified path
        """
        # Load trained model
        if settings.USE_MODEL:
            self._model = torch.load(
                path,
                map_location=self.device,
                weights_only=False,
            )
        else:
            self._model = SimpleRecommender()
            with torch.no_grad():
                self._model.linear.weight.fill_(1.0)
                self._model.linear.bias.fill_(1.0)
            self._model = torch.jit.script(self._model)

        self._model.eval()

        # TorchScript Optimization
        # self._model = torch.jit.freeze(self._model)

    def score(self, user_ids, playlist_ids):
        u = self._model.user_embeddings(user_ids)
        p = self._model.playlist_embeddings(playlist_ids)
        return (u * p).sum(1)

    def preprocess(self, user_ids: list[int]) -> torch.Tensor:
        """
        Convert Input to pytorch tensor
        """
        return (
            torch.tensor(user_ids, dtype=torch.float32)
            .reshape(len(user_ids), 1)
            .to(self.device)
        )
        # return torch.tensor([user_id], dtype=torch.float32).to(self.device)

    def predict(self, user_ids: list[int], top_k=5):
        all_user_ids = []
        all_playlist_ids = []
        slice_bounds = []  # (user_id, start_idx, end_idx)
        cold_user_no = 0
        for user_id in user_ids:
            playlists = user_to_playlists.get(user_id)

            if not playlists:
                # Fallback to mainstream user
                playlists = user_to_playlists.get(mainstream_user_id, [])
                effective_user_id = mainstream_user_id
                cold_user_no += 1
            else:
                effective_user_id = user_id

            if playlists:
                start = len(all_user_ids)
                all_user_ids.extend([effective_user_id] * len(playlists))
                all_playlist_ids.extend(playlists)
                end = len(all_user_ids)
                slice_bounds.append(
                    (user_id, start, end)
                )  # Keep original user_id for output
            else:
                slice_bounds.append((user_id, None, None))

        if not all_user_ids:
            return {user_id: [] for user_id in user_ids}

        user_tensor = torch.LongTensor(all_user_ids)
        playlist_tensor = torch.LongTensor(all_playlist_ids)

        with torch.inference_mode():
            scores = self.score(user_tensor, playlist_tensor)

        result = {}
        for user_id, start, end in slice_bounds:
            if start is None:
                result[user_id] = []
            else:
                user_scores = scores[start:end]
                user_playlists = all_playlist_ids[start:end]
                topk = min(top_k, len(user_scores))
                top_indices = torch.topk(user_scores, k=topk).indices
                result[user_id] = [user_playlists[i] for i in top_indices]

        return result, cold_user_no


pool = None
model = Recommender()


def intialize_model():
    global pool
    pool = ProcessPoolExecutor(
        max_workers=1, initializer=model.load_model(settings.MODEL_PATH)
    )
    print("Intialize Child Process for ML inference")


def make_prediction(user_id: list[int]) -> list[int]:
    if not settings.USE_MODEL:
        return {user: [user + i for i in range(5)] for user in user_id}, 0
    return model.predict(user_id)
