FROM nvcr.io/nvidia/nemo:latest

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir runpod

WORKDIR /app
COPY handler.py .

ENV PYTHONUNBUFFERED=1
ENV NEMO_CACHE_DIR=/cache/nemo-models

CMD ["python", "-u", "handler.py"]
