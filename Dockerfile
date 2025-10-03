# Streamlit app container built with uv-managed dependencies
FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

# Copy lockfiles first for caching
COPY pyproject.toml uv.lock ./

# Install dependencies into the project environment (creates .venv)
RUN uv sync --frozen --no-dev

# Copy application source
COPY . .

ENV PYTHONPATH=/app \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
