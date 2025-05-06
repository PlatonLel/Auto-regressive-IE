FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    git \
    libjsonnet-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/PlatonLel/Auto-regressive-IE.git /app

RUN pip install --no-cache-dir poetry==1.8.3

COPY pyproject.toml poetry.lock* ./

COPY checklist.tar.gz .
RUN rm -rf /app/.git \
    && poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi \
    && rm -rf ~/.cache/pypoetry
RUN rm -rf checklist.tar.gz

COPY checkpoints/best_checkpoint.pt checkpoints/best_checkpoint.pt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
