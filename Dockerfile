# Stock Market Prediction & Analysis System - research/educational use only, not financial advice.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# libgomp: OpenMP runtime needed by XGBoost and PyTorch on slim images
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch keeps the image far smaller than the default CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY configs ./configs
COPY .streamlit ./.streamlit
COPY src ./src
COPY api ./api
COPY app ./app
COPY main.py .

# Run as a non-root user; these folders are normally bind-mounted by docker-compose
RUN useradd --create-home --uid 1000 app \
    && mkdir -p data/raw data/processed models reports/figures \
    && chown -R app:app /app
USER app

EXPOSE 8000 8501

# Default: the API. AUTO_TRAIN=1 trains an XGBoost model first if none exists (needs internet).
# PORT is honoured so the same image works on platforms such as Render.
CMD ["sh", "-c", "python -m src.serving.bootstrap && uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
