FROM python:3.10-slim

# HF Spaces requires a non-root user
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# install python deps first (layer caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# copy project files
COPY --chown=user src/ src/
COPY --chown=user frontend/ frontend/
COPY --chown=user evaluation/ evaluation/
COPY --chown=user setup.py .
COPY --chown=user README.md .
COPY --chown=user .env.example .env

# writable dirs for data pipeline + logs
RUN mkdir -p /app/data/processed && chown -R user:user /app

USER user

ENV PORT=7860
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
