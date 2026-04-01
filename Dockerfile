FROM python:3.10-slim

WORKDIR /app

# install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source code
COPY src/ src/
COPY frontend/ frontend/
COPY setup.py .
COPY .env.example .env

# HF Spaces sets PORT=7860 by default
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# the data pipeline runs on first startup via the lifespan hook
# but we can also pre-run it in the build step for faster cold starts
# NOTE: this requires the OPENAI_API_KEY to be set as a build secret
# If not available, setup will run at first request instead

EXPOSE 7860

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
