import random
import json

from locust import HttpUser, task, between


class AppTest(HttpUser):
    wait_time = between(0, 2)

    @task
    def call_prediction(self):
        data = random.randint(0, 2000)
        self.client.post("api/playlist/recommend/", data=json.dumps({"user_id": data}))
