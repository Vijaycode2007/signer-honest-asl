# Build from the REPOSITORY ROOT, not from app/:
#   docker build -f app/Dockerfile -t sign-reader .
# The image needs both the app and the research package it imports, and Docker
# cannot copy from above the build context.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/app/.torch \
    HF_HOME=/app/.hf

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Dependencies first: they change far less often than the code, so this layer
# stays cached across ordinary deploys.
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The measured research package, imported rather than duplicated.
COPY backend/src/slr ./slr

COPY app/server ./server
COPY app/web ./web
COPY app/model_card.json ./model_card.json

# Weights are NOT baked in: 172 MB of fp16 checkpoint does not belong in an
# image or in git. The server fetches SLR_WEIGHTS_URL on first boot and caches
# it here.
RUN mkdir -p /app/weights
ENV SLR_WEIGHTS=/app/weights/deit3_base.pt

EXPOSE 8000
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
