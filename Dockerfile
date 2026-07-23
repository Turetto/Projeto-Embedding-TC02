# ── Stage 1: builder ──────────────────────────────────────
FROM python:3.12-slim AS builder
WORKDIR /app
RUN pip install uv
COPY pyproject.toml uv.lock README.md ./
RUN uv venv --seed .venv && uv sync --frozen --no-dev

# ── Stage 2: runtime ──────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv   
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY .env.example .env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

CMD ["bash", "scripts/entrypoint.sh"]