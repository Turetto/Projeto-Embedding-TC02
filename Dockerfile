# ── Stage 1: builder ──────────────────────────────────────
FROM python:3.14.0-slim AS builder

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# ── Stage 2: runtime ──────────────────────────────────────
FROM python:3.14.0-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
COPY configs/ ./configs/
COPY .env.example .env

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "scripts/validate_env.py"]