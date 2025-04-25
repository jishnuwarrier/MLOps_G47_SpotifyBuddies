import torch
import torch.nn as nn

from config import settings


# TODO => Remove after stable model is created
class SimpleRecommender(nn.Module):
    def __init__(self):
        super(SimpleRecommender, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


class Recommender:
    def __init__(self, model_path: str | None = None):
        """
        Initialize the model by loading the
        trained model from specified path
        """

        # Load trained model
        if settings.USE_MODEL:
            self.model = torch.load(model_path)
        else:
            self.model = SimpleRecommender()
            with torch.no_grad():
                self.model.linear.weight.fill_(1.0)
                self.model.linear.bias.fill_(1.0)

        self.model.eval()

    def preprocess(self, user_id: int) -> torch.Tensor:
        """
        Convert Input to pytorch tensor
        """
        return torch.tensor([[user_id]], dtype=torch.float32)

    def predict(self, user_id: int) -> int:
        """
        Make Prediction using the model with the input
        """

        # Preprocess the input
        input = self.preprocess(user_id)

        with torch.no_grad():
            recommendation = self.model(input).item()

        return recommendation
