from prometheus_client import Counter, Histogram


INFERENCE_COUNT = Counter("inference_requests_total", "Total inference requests")
COLD_USER = Counter("cold_user_total", "Total cold user encountered")
INFERENCE_TIME = Histogram(
    "inference_time_seconds",
    "Time spent on ML inference",
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5],
)
INFERENCE_DIVERSITY = Histogram(
    "inference_diversity",
    "Diversity of Predictions",
    buckets=[0, 1, 2, 5, 10, 20, 50, 100, 1000, 10000, 100000, 10000000],
)
REDIS_TO_TIME = Histogram(
    "redis_io_time_seconds",
    "Time spent on Redis I/O opeartions",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
)
