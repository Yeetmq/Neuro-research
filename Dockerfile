FROM nvidia/cuda:12.1-base

RUN apt-get update && apt-get install -y python3-pip

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5002

CMD ["uvicorn", "models.bart:app", "--host", "0.0.0.0", "--port", "5002"]