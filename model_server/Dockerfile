# ---------- builder ----------
FROM python:3.12-slim-bookworm AS builder

# (optional) only needed if you compile wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# install runtime deps
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# copy the rest of the application
ADD . /app

# ---------- final image ----------
# You can keep a single-stage image if size isn’t critical; otherwise:
# FROM python:3.12-slim-bookworm
# WORKDIR /app
# COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
# COPY --from=builder /app /app

EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
