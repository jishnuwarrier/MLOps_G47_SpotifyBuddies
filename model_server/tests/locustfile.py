import json

import numpy as np

from locust import HttpUser, task, between


class AppTest(HttpUser):
    wait_time = between(0, 2)

    @task
    def call_prediction(self):
        data = np.random.randint(1, 99999999, 1000)
        self.client.post(
            "api/playlist/recommend/", data=json.dumps({"user_ids": data.tolist()})
        )
