from prometheus_client import Counter, Histogram


INFERENCE_REQUESTS = Counter("inference_requests_total", "Total inference requests")
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds", "Inference latency in seconds"
)
