import requests
import re
from bs4 import BeautifulSoup
import certifi
from requests.exceptions import SSLError
from concurrent.futures import ThreadPoolExecutor, as_completed
import wikipedia


def wiki_search(query):
    wikipedia.set_lang("en")
    content = wikipedia.summary(query)
    return content

def load_sites(filename="utils/sites.txt"):
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

def filter_advertisements(soup):
    ads_classes = ["advertisement", "ad", "promo", "sponsored", "banner", "sidebar"]
    for ad_class in ads_classes:
        for ad_tag in soup.find_all(class_=ad_class):
            ad_tag.decompose() 

    ad_ids = ["ad-banner", "adblock", "ads", "popup"]
    for ad_id in ad_ids:
        ad_tag = soup.find(id=ad_id)
        if ad_tag:
            ad_tag.decompose()
    return soup

def extract_main_content(soup):
    content = ""
    main_content = soup.find(['article', 'main', 'section', 'div'], class_=re.compile(r'.*(content|body|main).*', re.IGNORECASE))
    if main_content:
        content = main_content.get_text(strip=True)
    else:
        for tag in ["p", "div", "span", "section"]:
            elements = soup.find_all(tag)
            for element in elements:
                content += element.get_text(strip=True) + "\n"
    return content

def is_relevant_page(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text().lower()
        return not any(keyword in text for keyword in ["курс", "платный", "course"])
    except:
        return False

def parse_webpage(session, url):
    try:
        if is_relevant_page(url):
            response = session.get(url, verify=certifi.where())
            
            if response.status_code != 200:
                print(f"Ошибка доступа к {url}: статус код {response.status_code}")
                return

            content = response.text
            
            if not is_valid_page(content):
                print(f"Страница с {url} содержит ошибку или проверку на робота.")
                return
            
            soup = BeautifulSoup(response.content, "html.parser")
            soup = filter_advertisements(soup) 
            
            title = soup.title.string if soup.title else "Без заголовка"
            print(f"Заголовок страницы: {title}")
            
            content = extract_main_content(soup) 
            
            if content:
                with open("utils/page_content.txt", "a", encoding="utf-8") as file:
                    file.write(f"\n\n{'='*50}\n")
                    file.write(f"Заголовок страницы: {title}\n")
                    file.write(f"URL: {url}\n")
                    file.write(f"{'='*50}\n\n")
                    
                    file.write(content)
                    
                    file.write("\n\n" + "="*50 + "\n\n")
                
                print(f"Содержимое страницы {url} записано в файл.")
            else:
                print(f"Контент не найден на странице {url}.")
        else: 
            print(f'invalid page (ad)')
    except SSLError as e:
        print(f"SSL ошибка при попытке соединения с {url}: {e}")
    except Exception as e:
        print(f"Ошибка при обработке {url}: {e}")

def parse_all_pages(sites):
    session = requests.Session()  
    with ThreadPoolExecutor(max_workers=10) as executor:  
        future_to_url = {executor.submit(parse_webpage, session, url): url for url in sites}
        for future in as_completed(future_to_url):
            future.result()  


if __name__ == '__main__':
    sites = load_sites('utils/sites.txt')
    print(f"Загружено {len(sites)} сайтов")
    video_links, webpage_links = classify_links(sites)

    parse_all_pages(webpage_links)

    with open("utils/page_content.txt", "a", encoding="utf-8") as file:
        file.write(wiki_search('transformers'))
