import sys
import os
import pandas as pd
import re
import unicodedata
import uuid

current_dir = os.path.dirname(os.path.abspath(__file__)) 
scripts_dir = os.path.dirname(current_dir)               
sys.path.insert(0, scripts_dir)                          

try:
    from config import DATA_DIR
except ImportError:
    DATA_DIR = "./data"

LABEL_MAPPING = {
    "Rối loạn tăng động giảm chú ý (ADHD)": {"class_name": "rối loạn tăng động giảm chú ý", "class_id": 0},
    "Rối loạn lo âu (Anxiety)": {"class_name": "rối loạn lo âu", "class_id": 1},
    "Rối loạn lưỡng cực (Bipolar)": {"class_name": "rối loạn lưỡng cực", "class_id": 2},
    "Trầm cảm (Depression)": {"class_name": "trầm cảm", "class_id": 3},
    "Rối loạn căng thẳng sau sang chấn (PTSD)": {"class_name": "rối loạn căng thẳng sau sang chấn", "class_id": 4}
}

KEYWORDS_DICT = {
    "Rối loạn tăng động giảm chú ý (ADHD)": [
        "rối loạn tăng động giảm chú ý", "chứng tăng động giảm chú ý", "tăng động giảm chú ý", "tăng động"
    ],
    "Rối loạn lo âu (Anxiety)": [
        "rối loạn lo âu", "lo âu", "chứng lo âu", "sự lo lắng", "nỗi lo âu", "bồn chồn", "bất an"
    ],
    "Rối loạn lưỡng cực (Bipolar)": [
        "rối loạn lưỡng cực", "bệnh hưng trầm cảm", "chứng rối loạn lưỡng cực", "bệnh lưỡng cực"
    ],
    "Trầm cảm (Depression)": [
        "bệnh trầm cảm", "chứng trầm cảm", "u uất", "sầu uất", "suy nhược tinh thần", "trầm cảm"
    ],
    "Rối loạn căng thẳng sau sang chấn (PTSD)": [
        "rối loạn căng thẳng sau chấn thương", "rối loạn căng thẳng sau sang chấn",
        "rối loạn stress sau sang chấn", "hậu chấn thương"
    ]
}

def assign_vietnamese_class(text_to_check):
    text_to_check = str(text_to_check).lower()
    for main_key, class_info in LABEL_MAPPING.items():
        if class_info['class_name'] in text_to_check:
            return class_info['class_name'], class_info['class_id']
        for keyword in KEYWORDS_DICT[main_key]:
            if keyword in text_to_check:
                return class_info['class_name'], class_info['class_id']
    return 'không liên quan đến các bệnh tâm lý', 5


def clean_text(text):
    if pd.isna(text) or str(text).strip() == "":
        return ""
    text = str(text)
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    
    vietnamese_chars = "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ"
    pattern = r'[^a-zA-Z0-9\s.!?,\-;:()\[\]{}"\'' + vietnamese_chars + r']'
    text = re.sub(pattern, '', text)
    
    text = re.sub(r'[\n\r\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_pullpush_pipeline():
    input_file = "dataPullpush_vi.xlsx"
    print(f" '{input_file}' ")
    try:
        df = pd.read_excel(input_file)
    except FileNotFoundError:
        return

    if 'ID' not in df.columns:
        df.insert(0, 'ID', [str(uuid.uuid4()) for _ in range(len(df))])

    label_col = None
    for col in ['Label', 'Subreddit(Label of category)', 'class_name', 'Subreddit']:
        if col in df.columns:
            label_col = col
            break
            
    if label_col is None:
        return

    initial_len = len(df)
    df[label_col] = df[label_col].fillna('')
    df = df[~df[label_col].astype(str).str.contains(',')].copy()


    df[['class_name', 'class_id']] = df.apply(
        lambda row: pd.Series(assign_vietnamese_class(str(row[label_col]))), axis=1
    )

    title_col = 'Title' if 'Title' in df.columns else 'title'
    content_col = 'Content' if 'Content' in df.columns else 'content'
    
    df['title_raw'] = df.get(title_col, pd.Series([''] * len(df))).fillna('')
    df['post_raw'] = df.get(content_col, pd.Series([''] * len(df))).fillna('')

    df['title'] = df['title_raw'].apply(clean_text)
    df['post'] = df['post_raw'].apply(clean_text)

    df['text'] = df.apply(lambda row: f"{row['title']}: {row['post']}" if row['title'] else row['post'], axis=1)
    
    df = df[df['text'].str.strip() != ''].copy()

    final_cols = ['ID', 'title', 'post', 'class_name', 'class_id', 'text']
    df_final = df[[col for col in final_cols if col in df.columns]]

    print(" (Class Distribution) ")
    print(df_final['class_name'].value_counts())

    save_dir = os.path.join(DATA_DIR, "processed_pullpush")
    os.makedirs(save_dir, exist_ok=True)

    output_path = os.path.join(save_dir, "pullpush_preprocessed_full.csv")
    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f" ({len(df_final)}  -> {output_path}")

if __name__ == "__main__":
    process_pullpush_pipeline()