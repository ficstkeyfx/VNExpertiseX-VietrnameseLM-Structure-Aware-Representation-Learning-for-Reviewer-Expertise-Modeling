import os
import csv
import time
import json
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_DIR = "d:/C500/Lab306/Reviewer_Recommendation/crawl-data"
LINKS_FILE = os.path.join(BASE_DIR, "crawled_links.txt")
DATA_FILE = os.path.join(BASE_DIR, "articles_data.csv")
PROGRESS_FILE = os.path.join(BASE_DIR, "extractor_progress.txt")
ARTICLES_DIR = os.path.join(BASE_DIR, "articles")

os.makedirs(ARTICLES_DIR, exist_ok=True)

# Map Vietnamese labels to English keys
FIELD_MAPPING = {
    "Lĩnh vực nghiên cứu": "research_field",
    "Tác giả": "author",
    "Nhan đề": "title",
    "Nhan đề tiếng anh": "english_title",
    "Nguồn trích": "source",
    "Năm xuất bản": "publish_year",
    "Số": "issue",
    "Trang": "pages",
    "ISSN": "issn",
    "Từ khóa": "keywords",
    "Từ khóa tiếng anh": "english_keywords",
    "Tóm tắt": "abstract",
    "Tóm tắt tiếng anh": "english_abstract",
    "Link file pdf toàn văn": "pdf_link",
    "Tên file PDF cục bộ": "local_pdf_file"
}

# The keys in English that will be in our CSV and JSON
FIELDS_TO_EXTRACT = list(FIELD_MAPPING.values())

def init_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--headless=new")

    prefs = {
        "download.default_directory": os.path.abspath(ARTICLES_DIR),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True
    }
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def load_processed_urls():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_progress(url):
    with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
        f.write(url + '\n')

def extract_id_from_url(url):
    match = re.search(r'-(\d+)\.html$', url)
    if match:
        return match.group(1)
    return str(abs(hash(url)))

def extract_data(driver, url, article_id):
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    
    data = {"url": url}
    for f in FIELDS_TO_EXTRACT:
        data[f] = ""

    labels = soup.find_all("label", class_="control-label")
    for label in labels:
        l_text = label.text.strip().lower()
        matched_en_key = None
        
        # Exact match (case-insensitive) is safer to avoid partial word matching like "số" in another word
        for vi_key in FIELD_MAPPING.keys():
            if vi_key.lower() == l_text.strip().lower():
                matched_en_key = FIELD_MAPPING[vi_key]
                break
        
        if matched_en_key:
            value_div = label.find_next_sibling("div")
            if value_div:
                val = value_div.text.strip().replace('\n', ' ')
                val = re.sub(r'\s+', ' ', val)
                data[matched_en_key] = val
                
    if not data.get("title"):
        title_tag = soup.find('h1') or soup.find('title')
        if title_tag:
             data["title"] = title_tag.text.strip()

    # Find PDF Links and Download
    pdf_tag = soup.find('a', string=re.compile(r'Toàn văn|PDF', re.IGNORECASE))
    if not pdf_tag:
        pdf_tag = soup.find('a', href=re.compile(r'download|pdf', re.IGNORECASE))
        
    local_pdf_name = ""
    if pdf_tag:
        pdf_href = pdf_tag.get('href', '')
        data["pdf_link"] = pdf_href
        
        try:
            elements = driver.find_elements(By.XPATH, "//a[contains(@href, 'download') or contains(@href, 'pdf')]")
            target_el = None
            for el in elements:
                if "toàn văn" in el.text.lower() or "pdf" in el.text.lower() or not el.text:
                    target_el = el
                    break
            if not target_el and elements:
                target_el = elements[0]
                
            if target_el:
                existing_files = set(os.listdir(ARTICLES_DIR))
                
                driver.execute_script("arguments[0].click();", target_el)
                print("   -> Đang tải PDF...")
                
                downloaded_file = None
                for _ in range(30):
                    time.sleep(1)
                    current_files = set(os.listdir(ARTICLES_DIR))
                    new_files = current_files - existing_files
                    
                    is_downloading = any(f.endswith('.crdownload') or f.endswith('.tmp') for f in new_files)
                    
                    if new_files and not is_downloading:
                        for f in new_files:
                            if not f.endswith('.crdownload') and not f.endswith('.tmp'):
                                downloaded_file = f
                                break
                    if downloaded_file:
                        break
                
                if downloaded_file:
                    ext = os.path.splitext(downloaded_file)[1]
                    new_name = f"article_{article_id}{ext}"
                    old_path = os.path.join(ARTICLES_DIR, downloaded_file)
                    new_path = os.path.join(ARTICLES_DIR, new_name)
                    if os.path.exists(new_path):
                        os.remove(new_path)
                    os.rename(old_path, new_path)
                    local_pdf_name = new_name
                    print(f"   -> Đã tải xong PDF: {local_pdf_name}")
                else:
                    print("   -> Lỗi: Tải PDF quá lâu (Timeout)")
                    
        except Exception as e:
            print(f"   -> Lỗi khi click tải PDF: {e}")
            
    data["local_pdf_file"] = local_pdf_name
    
    # Save JSON file
    json_path = os.path.join(ARTICLES_DIR, f"article_{article_id}.json")
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(data, jf, ensure_ascii=False, indent=4)
        print(f"   -> Đã lưu JSON: {json_path}")

    return data

def main():
    if not os.path.exists(LINKS_FILE):
        print(f"Không tìm thấy {LINKS_FILE}. Vui lòng chạy crawler.py trước.")
        return

    links_to_process = []
    with open(LINKS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split(" | ", 1)
            if len(parts) > 0:
                url = parts[0].strip()
                if url:
                    links_to_process.append(url)

    processed_urls = load_processed_urls()
    pending_urls = [u for u in links_to_process if u not in processed_urls]

    print(f"Tổng số link bài báo: {len(links_to_process)}")
    print(f"Số link đã xử lý: {len(processed_urls)}")
    print(f"Số link cần xử lý: {len(pending_urls)}")

    if not pending_urls:
        print("Đã xử lý tất cả các link hiện có!")
        return

    file_exists = os.path.exists(DATA_FILE)
    csv_file = open(DATA_FILE, 'a', newline='', encoding='utf-8')
    fieldnames = ["url"] + FIELDS_TO_EXTRACT
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    if not file_exists:
        writer.writeheader()

    driver = None
    try:
        driver = init_driver()
        for i, url in enumerate(pending_urls, 1):
            article_id = extract_id_from_url(url)
            print(f"[{i}/{len(pending_urls)}] Đang cào dữ liệu từ: {url}")
            try:
                driver.get(url)
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Tác giả') or contains(text(), 'Lĩnh vực')]"))
                    )
                except Exception:
                    pass 
                
                time.sleep(1) 
                data = extract_data(driver, url, article_id)
                
                writer.writerow(data)
                csv_file.flush()
                save_progress(url)
                
            except Exception as e:
                print(f"   -> Lỗi khi xử lý {url}: {e}")
            
            time.sleep(1.5)
            
    finally:
        csv_file.close()
        if driver:
            driver.quit()

if __name__ == "__main__":
    main()
