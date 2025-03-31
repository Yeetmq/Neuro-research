import requests
import re
from bs4 import BeautifulSoup
import certifi
from requests.exceptions import SSLError

def load_sites(filename="sites.txt"):
    try:
        with open(filename, "r") as file:
            return [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        return [] 
    


def classify_links(links):
    video_links = []
    webpage_links = []

    for link in links:
        if re.search(r"(youtube\.com|rutube\.ru|vimeo\.com)", link, re.IGNORECASE):
            video_links.append(link)  
        else:
            webpage_links.append(link)  
    return video_links, webpage_links

def is_valid_page(content):
    if "Вы не робот?" in content or "Access forbidden" in content:
        return False
    return True

def parse_webpage(url):
    try:
        response = requests.get(url, verify=certifi.where())
        
        if response.status_code != 200:
            print(f"Ошибка доступа к {url}: статус код {response.status_code}")
            return

        content = response.text
        
        if not is_valid_page(content):
            print(f"Страница с {url} содержит ошибку или проверку на робота.")
            return
        
        soup = BeautifulSoup(response.content, "html.parser")
        title = soup.title.string if soup.title else "Без заголовка"
        print(f"Заголовок страницы: {title}")
        
        paragraphs = soup.find_all("p")
        page_content = "\n".join([p.get_text() for p in paragraphs]) 
        
        with open("page_content.txt", "a", encoding="utf-8") as file:
            file.write(f"Содержимое страницы {url}:\n")
            file.write(page_content)
            file.write("\n\n---\n\n")
        
        print(f"Содержимое страницы {url} записано в файл.")
        
    except SSLError as e:
        print(f"SSL ошибка при попытке соединения с {url}: {e}")
    except Exception as e:
        print(f"Ошибка при обработке {url}: {e}")

sites = load_sites('sites.txt')

video_links, webpage_links = classify_links(sites)

for url in webpage_links:
    parse_webpage(url)