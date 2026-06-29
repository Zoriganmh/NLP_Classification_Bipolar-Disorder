import praw
import datetime
import pandas as pd
import os
import re

reddit = praw.Reddit(
    client_id="your_client_id",
    client_secret="your_client_secret",
    user_agent="your_user_agent"
)

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

all_posts = []

for subreddit_name in subreddits:
    print(f" r/{subreddit_name} ...")
    subreddit = reddit.subreddit(subreddit_name)

    matched_count = 0
    for submission in subreddit.new(limit=None):
        title = submission.title or ""
        content = submission.selftext or ""
        
        full_text = f"{title} {content}"
        label = get_labels(full_text)
        
        if label:
            author = submission.author.name if submission.author else "[deleted]"
            created_time = datetime.datetime.fromtimestamp(submission.created_utc)

            all_posts.append({
                "Author": author,
                "Posted at": created_time.strftime("%Y-%m-%d %H:%M:%S"),
                "Title": title,
                "Content": content,
                "Subreddit": subreddit_name,
                "Label": label
            })
            matched_count += 1
            
    print(f" {matched_count} of r/{subreddit_name}.")

output_folder = "data"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "dataTiengVietPRAW.xlsx")

df = pd.DataFrame(all_posts)

def clean_excel_string(s):
    if isinstance(s, str):
        return re.sub(r'[\x00-\x1F]+', '', s)
    return s


df = df.map(clean_excel_string)
df.to_excel(output_file, index=False)

print(f"\n {len(all_posts)}")
print(f" {os.path.abspath(output_file)}")