import requests
import pandas as pd
import datetime
import os
import time
import re
import csv

subreddits = ["vozforums", "TroChuyenLinhTinh"]

KEYWORDS_DICT = {
    "Rối loạn tăng động giảm chú ý (ADHD)": [
        "rối loạn tăng động giảm chú ý", "chứng tăng động giảm chú ý", 
        "hội chứng tăng động giảm chú ý", "tăng động giảm chú ý", "tăng động"
    ],
    "Rối loạn lo âu (Anxiety)": [
        "rối loạn lo âu", "lo âu", "chứng lo âu", "sự lo lắng", 
        "nỗi lo âu", "trạng thái bồn chồn", "bất an"
    ],
    "Rối loạn lưỡng cực (Bipolar)": [
        "rối loạn lưỡng cực", "bệnh hưng trầm cảm", 
        "chứng rối loạn lưỡng cực", "bệnh lưỡng cực"
    ],
    "Trầm cảm (Depression)": [
        "bệnh trầm cảm", "chứng trầm cảm", "u uất", "sầu uất", 
        "suy nhược tinh thần", "trầm cảm"
    ],
    "Rối loạn căng thẳng sau sang chấn (PTSD)": [
        "rối loạn căng thẳng sau chấn thương", "rối loạn căng thẳng sau sang chấn",
        "rối loạn stress sau sang chấn", "hội chứng chấn thương tâm lý", 
        "rối loạn tâm lý sau sang chấn", "căng thẳng hậu chấn thương"
    ]
}

def get_labels(text):
    if not text:
        return None
    text_lower = text.lower()
    matched_labels = []
    
    for label, keywords in KEYWORDS_DICT.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                if label not in matched_labels:
                    matched_labels.append(label)
                break
                
    return ", ".join(matched_labels) if matched_labels else None

def clean_excel_string(s):
    if isinstance(s, str):
        return re.sub(r'[\x00-\x1F]+', '', s).replace('\n', ' ')
    return s

def safe_get(url, params, retries=8, timeout=20):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    for i in range(retries):
        try:
            res = requests.get(url, params=params, headers=headers, timeout=timeout)
            if res.status_code == 200:
                return res
            elif res.status_code == 429:
                wait_time = 15 * (2 ** i) 
                print(f"\n  {wait_time} ")
                time.sleep(wait_time)
            else:
                print(f" {res.status_code}, ({i+1}/{retries}) ...")
                time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f" ({i+1}/{retries}): {e}")
            time.sleep(5)
    return None

# Time
start_date = datetime.datetime(2019, 1, 1)
end_date = datetime.datetime(2020, 12, 31)
step = datetime.timedelta(days=3)
url = "https://api.pullpush.io/reddit/search/submission/"

os.makedirs("data", exist_ok=True)
output_file = os.path.join("data", "dataTiengViet_Filtered1.csv")
file_exists = os.path.isfile(output_file)

total_matched = 0
current = start_date

with open(output_file, mode='a', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["Author", "Posted at", "Title", "Content", "Subreddit", "Label", "Date Range"])
    
    if not file_exists:
        writer.writeheader() 

    while current < end_date:
        after = int(current.timestamp())
        before = int((current + step).timestamp())
        interval_label = f"{current.strftime('%Y-%m-%d')} → {(current + step).strftime('%Y-%m-%d')}"
        
        print(f"\n {interval_label}")

        for sub in subreddits:
            params = {
                "subreddit": sub,
                "after": after,
                "before": before,
                "size": 100
            }

            res = safe_get(url, params)
            if res:
                data = res.json().get("data", [])
                print(f"  {len(data)} of r/{sub}. ")
                
                matched_count = 0
                for post in data:
                    title = clean_excel_string(post.get("title", ""))
                    content = clean_excel_string(post.get("selftext", ""))
                    
                    full_text = f"{title} {content}"
                    label = get_labels(full_text)
                    
                    if label: 
                        row_data = {
                            "Author": clean_excel_string(post.get("author")),
                            "Posted at": datetime.datetime.fromtimestamp(int(post["created_utc"])).strftime("%Y-%m-%d %H:%M:%S"),
                            "Title": title,
                            "Content": content,
                            "Subreddit": sub,
                            "Label": label,
                            "Date Range": interval_label
                        }
                        writer.writerow(row_data)
                        f.flush() 
                        matched_count += 1
                        total_matched += 1
                        
                print(f" final {matched_count} ")
            else:
                print(f"  r/{sub} of {interval_label}")
            
            time.sleep(1.5)

        current += step

print(f" total: {total_matched}")
print(f" {os.path.abspath(output_file)}")

try:
    df_final = pd.read_csv(output_file)
    excel_path = output_file.replace('.csv', '.xlsx')
    df_final.to_excel(excel_path, index=False)
    print(f" {os.path.abspath(excel_path)}")
except Exception as e:
    print(f" {e} ")