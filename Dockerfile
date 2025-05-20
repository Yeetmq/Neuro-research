FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-setuptools \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip install torch==2.3.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
RUN pip install -r requirements.txt

COPY . .

ENV MODEL_PATH=/models/bart
ENV DATA_DIR=/data
ENV CONFIG_PATH=/config/settings.yaml

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]