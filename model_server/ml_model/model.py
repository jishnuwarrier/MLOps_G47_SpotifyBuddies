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
            self._model = torch.jit.load(path, map_location=self.device)
        else:
            self._model = SimpleRecommender()
            with torch.no_grad():
                self._model.linear.weight.fill_(1.0)
                self._model.linear.bias.fill_(1.0)
            self._model = torch.jit.script(self._model)

        self._model.eval()

        # TorchScript Optimization
        self._model = torch.jit.freeze(self._model)

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

    def predict(self, user_ids: list[int]) -> dict[int, list[int]]:
        """
        Make Prediction using the model with the input
        """

        # Preprocess the input
        input = self.preprocess(user_ids)

        with torch.inference_mode():
            recommendations = self._model(input).reshape(-1).cpu().tolist()
            if not settings.USE_MODEL:
                recommendations = [[rec] for rec in recommendations]
            # recommendations = self._model(input).cpu().item()

        # Create a dictionary of user_ids as key and playlist_id as value
        results = {
            user_id: playlist_id
            for user_id, playlist_id in zip(user_ids, recommendations)
        }

        return results


pool = None
model = Recommender()


def intialize_model():
    global pool
    pool = ProcessPoolExecutor(
        max_workers=1, initializer=model.load_model(settings.MODEL_PATH)
    )
    print("Intialize Child Process for ML inference")


def make_prediction(user_id: [int]) -> list[int]:
    return model.predict(user_id)
