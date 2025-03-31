from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import requests
from bs4 import BeautifulSoup
import time

def get_yandex_results(query):
    chrome_options = Options()
    chrome_options.add_argument("--headless")  
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")  
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    chrome_options.add_argument("--lang=ru-RU")
    
    driver_path = "C:/Neuro-research/Neuro-research/chromedriver-win64/chromedriver.exe"

    service = Service(driver_path)

    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        url = f"https://yandex.ru/search/?text={query.replace(' ', '+')}"
        driver.get(url)
        
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-cid]"))
        )
        
        last_height = driver.execute_script("return document.body.scrollHeight")
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        links = []
        results = driver.find_elements(By.CSS_SELECTOR, "a.Link_theme_outer[href]")
        
        for link in results:
            href = link.get_attribute("href")
            if href and not href.startswith("https://yandex.ru"):  
                links.append(href)
                
        return list(set(links))[:20]
        
    except Exception as e:
        print(f"Ошибка: {e}")
        return []
    finally:
        driver.quit()  

sites = get_yandex_results("Машинное обучение и анализ данных")
print(f"Найдено {len(sites)} сайтов:")
for i, url in enumerate(sites, 1):
    print(f"{i}. {url}")

def save_sites(sites, filename="sites.txt"):
    with open(filename, "w") as file:
        for site in sites:
            file.write(site + "\n")

save_sites(sites)