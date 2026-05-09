FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Install NeMo stable (2.0.0 — не nightly, не использует nn.Buffer)
# Затем force-reinstall torch с CUDA-wheel чтобы вернуть GPU-версию torchvision
RUN pip install --no-cache-dir \
    "nemo_toolkit[asr]==2.0.0" \
    runpod \
    omegaconf && \
    pip install --no-cache-dir --force-reinstall \
    "torch==2.4.0" \
    "torchvision==0.19.0" \
    "torchaudio==2.4.0" \
    --index-url https://download.pytorch.org/whl/cu124

# nn.Buffer — type annotation требующая PyTorch 2.5+, нам не нужна (diarization не использует context biasing)
RUN find /usr/local/lib -name "biasing_multi_model.py" -exec \
    sed -i 's/nn\.Buffer | nn\.Parameter/nn.Parameter/g' {} \;

WORKDIR /app
COPY handler.py .

ENV PYTHONUNBUFFERED=1
ENV NEMO_CACHE_DIR=/cache/nemo-models

CMD ["python", "-u", "handler.py"]
