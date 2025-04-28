from .model import model


# TODO => Improve it after the stable model is created
def get_recommended_playlist(user_id: int) -> int:
    return model.predict(user_id)
