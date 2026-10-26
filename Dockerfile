FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /bench
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY suite.py run.py ./
ENV PATH="/bench/.venv/bin:$PATH"
