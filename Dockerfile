# Single image used for the API, the bootstrap job, and the retrainer (see docker-compose.yml).
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU-only torch wheel first to keep the image small and avoid the CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs

# Core + Postgres driver (the compose path uses Postgres + Redis).
RUN pip install --no-cache-dir -e ".[postgres]"

EXPOSE 8000

# Default: serve. Overridden per-service in docker-compose.yml.
CMD ["rtb", "serve"]
