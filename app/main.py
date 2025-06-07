from fastapi import FastAPI, Depends, HTTPException, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
import uvicorn
import yaml
import logging
import torch

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline_executor import execute_pipeline
from models.loader import load_pipeline, load_models


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Neuro-Research Pipeline", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="app/templates")

def load_config():
    config_path = os.getenv("CONFIG_PATH", "/app/cfg.yaml")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Файл конфигурации не найден по пути: {config_path}")
        raise

global_pipeline = None
global_summarizer = None
global_generator = None

@app.on_event("startup")
async def startup_event():
    """Инициализация моделей при старте сервера"""
    global global_summarizer, global_generator
    try:
        config = load_config()
        global_summarizer, global_generator = load_models(
            config['path_to_bart_model'], 
            config['llm_model_name']
        )
        logger.info("Модели загружены при старте")
    except Exception as e:
        logger.error(f"Ошибка загрузки моделей: {e}")
        raise RuntimeError("Не удалось загрузить модели при старте")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Главная страница с формой"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/run")
async def run_pipeline(
    request: Request,
    query: str = Form(...),
    links_num: int = Form(5)
):
    global global_summarizer, global_generator, last_results
    
    torch.cuda.empty_cache()
    logger.info("GPU-кэш очищен")
    
    if not global_summarizer or not global_generator:
        logger.error("Модели не загружены")
        raise HTTPException(status_code=500, detail="Модели не загружены")

    try:
        config = load_config()
        logger.info("CFG")

        # Выполняем пайплайн
        results = execute_pipeline(
            summarizer=global_summarizer,
            generator=global_generator,
            query=query,
            config_path=config["config_path"],
            data_dir=config["data_dir"],
            links_num=links_num
        )
        logger.info("Pipe")

        last_results = results

        torch.cuda.empty_cache()
        logger.info("GPU-кэш очищен")

        # Перенаправляем на /results
        return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "results": last_results
        }
    )
    
    except Exception as e:
        logger.error(f"Ошибка выполнения пайплайна: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results")
async def get_results(request: Request):
    """Отображение результатов пайплайна"""
    if last_results is None:
        return templates.TemplateResponse("no_results.html", {"request": request})

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "results": last_results
        }
    )