import os
import json
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_DIR = "d:/C500/Lab306/Reviewer_Recommendation/crawl-data"
ARTICLES_DIR = os.path.join(BASE_DIR, "articles")

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

def redownload_pdf(driver, url, article_id):
    try:
        driver.get(url)
        # Bỏ qua chờ nếu trang load chậm, nhưng chờ ít nhất một phần
        time.sleep(2)
        
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
            print(f"[{article_id}] Đang tải lại PDF...")
            
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
                print(f"[{article_id}] Đã tải xong PDF: {new_name}")
                return new_name
            else:
                print(f"[{article_id}] Lỗi: Tải PDF quá lâu (Timeout)")
                return None
        else:
            print(f"[{article_id}] Không tìm thấy nút tải PDF trên trang.")
            return None
    except Exception as e:
        print(f"[{article_id}] Lỗi khi tải lại PDF: {e}")
        return None

def main():
    print("Bắt đầu quét thư mục articles...")
    missing_pdfs = []
    
    # Lấy danh sách tất cả các file JSON
    json_files = [f for f in os.listdir(ARTICLES_DIR) if f.endswith('.json')]
    
    for jf_name in json_files:
        article_id = jf_name.replace('article_', '').replace('.json', '')
        pdf_name = f"article_{article_id}.pdf"
        pdf_path = os.path.join(ARTICLES_DIR, pdf_name)
        
        # Kiểm tra xem PDF có tồn tại chưa
        if not os.path.exists(pdf_path):
            missing_pdfs.append((jf_name, article_id))
            
    if not missing_pdfs:
        print("Tuyệt vời! Tất cả các file JSON đều đã có file PDF tương ứng.")
        return
        
    print(f"Phát hiện {len(missing_pdfs)} bài báo thiếu file PDF. Bắt đầu tiến trình tải lại...")
    
    driver = init_driver()
    try:
        for idx, (jf_name, article_id) in enumerate(missing_pdfs, 1):
            print(f"--- Tiến độ: {idx}/{len(missing_pdfs)} ---")
            json_path = os.path.join(ARTICLES_DIR, jf_name)
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            url = data.get('url')
            if not url:
                print(f"[{article_id}] Lỗi: Không tìm thấy URL trong file JSON.")
                continue
                
            new_pdf_name = redownload_pdf(driver, url, article_id)
            
            # Cập nhật lại JSON nếu tải thành công
            if new_pdf_name:
                data['local_pdf_file'] = new_pdf_name
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                    
    finally:
        driver.quit()
        print("Hoàn thành quá trình quét và tải lại.")

if __name__ == "__main__":
    main()
