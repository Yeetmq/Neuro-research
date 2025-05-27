docker build -t neuro-research .

if [ $? -ne 0 ]; then
    echo "Ошибка: Сборка Docker-образа не удалась."
    exit 1
fi

docker run --gpus all -d --rm \
  -p 8001:8001 \
  -v ./models:/app/models \
  -v ./data:/app/data \
  -v ./summarization_parser:/app/summarization_parser \
  -v ./final_model:/app/final_model \
  neuro-research

if [ $? -eq 0 ]; then
    echo -e "Доступно по адресу: https://192.168.202.60:9001/proxy/8001/\n"
else
    echo "Не удалось запустить контейнер."
    exit 1
fi