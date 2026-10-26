FROM python:3.12-slim
# torch.compile generates and builds C++ on CPU.
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /bench
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY models ./models
COPY run.py ./
ENV PATH="/bench/.venv/bin:$PATH"
