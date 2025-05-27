from fastapi import FastAPI, Depends, HTTPException, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
import uvicorn
import yaml
import logging

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline_executor import execute_pipeline


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

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

def load_config():
    config_path = os.getenv("CONFIG_PATH", "/app/cfg.yaml")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Файл конфигурации не найден по пути: {config_path}")
        raise

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Главная страница с формой"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/run")
async def run_pipeline(
    request: Request,
    query: str = Form(...),
    max_retries: int = Form(3),
    delay: int = Form(5)
):
    """Запуск пайплайна по поисковому запросу"""
    try:
        config = load_config()
        results = execute_pipeline(
            query=query,
            config_path=config["config_path"],
            data_dir=config["data_dir"],
            max_retries=max_retries,
            delay=delay
        )
        return templates.TemplateResponse(
        "results.html", 
        {
            "request": request,
            "results": results
        })
    except Exception as e:
        logger.error(f"Ошибка выполнения пайплайна: {e}")
        raise HTTPException(status_code=500, detail=str(e))