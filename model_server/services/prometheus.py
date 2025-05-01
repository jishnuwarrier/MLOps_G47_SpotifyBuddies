from prometheus_client import Counter, Histogram


INFERENCE_COUNT = Counter("inference_requests_total", "Total inference requests")
INFERENCE_TIME = Histogram(
    "inference_time_seconds",
    "Time spent on ML inference",
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5],
)
INFERENCE_COUNTER_OUPUT = Counter(
    "infernce_ouput_total",
    "Count of inference outputs by type",
    ["output_label"],
)
REDIS_TO_TIME = Histogram(
    "redis_io_time_seconds",
    "Time spent on Redis I/O opeartions",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
)
