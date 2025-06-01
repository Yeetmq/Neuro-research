FROM nvidia/cuda:12.8.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-setuptools \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*


ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_CACHE=$HF_HOME
ENV HUGGINGFACE_HUB_CACHE=$HF_HOME

WORKDIR /app
COPY req.txt .

RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir
RUN pip install -r req.txt --no-cache-dir

COPY . .


ENV MODEL_PATH='/final_model'
ENV DATA_DIR='/summarization_parser/data'
ENV CONFIG_PATH='/app/cfg.yaml'


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--timeout-keep-alive", "600", "--timeout-graceful-shutdown", "600"]