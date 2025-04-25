from .model import Recommender


# TODO => Improve it after the stable model is created
def get_recommended_playlist(user_id: int) -> int:
    recommender = Recommender()
    return recommender.predict(user_id)
