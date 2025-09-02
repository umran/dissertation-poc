FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    build-essential git curl ca-certificates nano \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

COPY pyproject.toml ./
COPY uv.lock* ./

RUN uv sync

COPY . .

RUN uv sync
RUN uv pip install -e .

CMD ["/bin/bash"]