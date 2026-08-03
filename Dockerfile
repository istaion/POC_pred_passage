# syntax=docker/dockerfile:1
FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Le binaire du venv est appelé directement (pas "uv run") pour éviter toute
# re-résolution des dépendances (dont le groupe dev) au démarrage du conteneur.
CMD ["sh", "-c", ".venv/bin/streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
