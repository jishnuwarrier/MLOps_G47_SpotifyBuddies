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

    def preprocess(self, user_id: int) -> torch.Tensor:
        """
        Convert Input to pytorch tensor
        """
        return torch.tensor([[user_id]], dtype=torch.float32).to(self.device)

    def predict(self, user_id: int) -> int:
        """
        Make Prediction using the model with the input
        """

        # Preprocess the input
        input = self.preprocess(user_id)

        with torch.inference_mode():
            recommendation = self._model(input).cpu().item()

        return recommendation


pool = None


def intialize_model():
    # global model
    # model.load_model(settings.MODEL_PATH)
    global pool
    pool = ProcessPoolExecutor(
        max_workers=1, initializer=Recommender().load_model(settings.MODEL_PATH)
    )
    print("Intialize Child Process for ML inference")


model = Recommender()


def make_prediction(user_id: int) -> int:
    return model.predict(user_id)
