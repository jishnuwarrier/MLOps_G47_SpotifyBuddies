    FROM python:3.11-slim AS build

    RUN apt-get update && \
        apt-get install -y --no-install-recommends build-essential git && \
        rm -rf /var/lib/apt/lists/*
    
    WORKDIR /src
    
    COPY model_server/pyproject.toml model_server/uv.lock ./model_server/
    
    RUN pip install --upgrade pip poetry && \
        poetry config virtualenvs.create false && \
        poetry install --no-interaction --only main --no-root
    
    FROM python:3.11-slim
    
    ENV USER=appuser
    RUN useradd -m $USER
    WORKDIR /app
    USER $USER
    
    COPY --from=build /usr/local/lib/python*/site-packages /usr/local/lib/python*/site-packages
    COPY model_server /app
    
    ENV PORT=8000
    EXPOSE $PORT
    
    HEALTHCHECK --interval=30s --timeout=3s CMD \
      wget -qO- http://localhost:${PORT}/healthz || exit 1
    

    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]