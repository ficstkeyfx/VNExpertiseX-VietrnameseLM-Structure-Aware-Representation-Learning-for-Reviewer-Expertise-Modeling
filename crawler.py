import sys
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Mức cấu hình
START_URL = "https://sti.vista.gov.vn/?mod=publication&keyword=&year=0&ts=all&af=&nb=4&cat=&sort=&page=1"
PROGRESS_FILE = "crawl_progress.txt"
LINKS_FILE = "crawled_links.txt"

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            url = f.read().strip()
            if url:
                return url
    return START_URL

def save_progress(url):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        f.write(url)

def save_links(links):
    with open(LINKS_FILE, 'a', encoding='utf-8') as f:
        for href, title in links:
            # Làm sạch title
            clean_title = title.replace('\n', ' ').replace('\r', '').strip()
            f.write(f"{href} | {clean_title}\n")

def init_driver():
    options = webdriver.ChromeOptions()
    # Thêm options cơ bản để vượt bot detection
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    # Tùy chọn chạy ẩn
    # options.add_argument("--headless")
    
    # Bạn cần cài đặt thư viện webdriver-manager hoặc chỉ định đường dẫn chrome driver.
    # Trong môi trường Python hiện đại, Selenium 4 có thể tự quản lý Driver.
    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

def main():
    driver = None
    try:
        driver = init_driver()
        current_url = load_progress()
        print(f"[*] Bắt đầu cào dữ liệu từ: {current_url}")
        driver.get(current_url)
        
        # Đợi chút cho WAF check qua (thường script cần đợi 3-5s nếu có challenge)
        time.sleep(3)
        
        while True:
            print(f"[*] Đang xử lý trang: {driver.current_url}")
            save_progress(driver.current_url)
            
            # 1. Trích xuất tất cả các link bài báo trên trang hiện tại
            # Các link chi tiết thường có dạng chứa mod=publication và fun=detail
            article_links = []
            
            # Đợi đến khi danh sách hoặc link xuất hiện (Tránh lỗi chưa render xong)
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '/publication/view')]"))
                )
            except TimeoutException:
                print("[-] Không tìm thấy link bài viết nào hoặc tải quá lâu.")
                pass
            
            elements = driver.find_elements(By.XPATH, "//a[contains(@href, '/publication/view')]")
            for el in elements:
                href = el.get_attribute('href')
                text = el.text.strip()
                if href and text:
                    article_links.append((href, text))
            
            # Loại bỏ link trùng lặp trên cùng 1 trang (đôi khi có ảnh và tiêu đề cùng link)
            unique_links = list({link[0]: link for link in article_links}.values())
            
            if unique_links:
                save_links(unique_links)
                print(f"[+] Đã lưu {len(unique_links)} bài báo.")
            else:
                print("[-] Trang này không có bài báo nào.")
            
            # 2. Tìm và click nút "Sau" (Next Page)
            try:
                # Tìm thẻ a có chữ "Sau" hoặc class/id tương tự biểu thị trang tiếp theo
                # Dựa vào HTML crawler trước: <a href="?mod...">Sau</a>
                next_btn = driver.find_element(By.XPATH, "//a[contains(text(), 'Sau') or contains(text(), 'Next')]")
                
                next_url = next_btn.get_attribute('href')
                print(f"[*] Chuyển sang trang tiếp theo: {next_url}")
                
                # Chuyển trang
                driver.get(next_url)
                
                # Tạm nghỉ 1 chút để tránh spam server
                time.sleep(2)
                
            except NoSuchElementException:
                print("[*] Không tìm thấy nút 'Sau'. Quá trình cào dữ liệu kết thúc.")
                break
                
    except Exception as e:
        print(f"[!] Lỗi không mong đợi: {e}")
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    main()
