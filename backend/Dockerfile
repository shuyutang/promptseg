# GPU image for the backend. Weights are not baked in: they are downloaded on
# first start into a mounted cache, so changing SAM2_MODEL_ID needs no rebuild.
# Build and run it with ./start_server.sh.
FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04

# System libs to decode JPEG/JPEG2000 in DICOMs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv libopenjp2-7 libjpeg-turbo8 && \
    rm -rf /var/lib/apt/lists/*

# Dedicated venv (avoids PEP 668 issues)
ENV VENV=/opt/venv
RUN python3 -m venv $VENV
ENV PATH="$VENV/bin:$PATH"

WORKDIR /app

# Torch first, from the CUDA wheel index (cu128 wheels run fine on a 12.9 runtime).
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
# Keep weights on a mounted volume so the image does not have to be rebuilt
# to change models, and the download survives container restarts.
ENV HF_HOME=/weights
ENV SAM2_MODEL_ID=facebook/sam2.1-hiera-base-plus

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
