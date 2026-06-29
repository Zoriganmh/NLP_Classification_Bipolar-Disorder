import sys
import os
import emoji
import pandas as pd
import re
import unicodedata
import uuid
from sklearn.model_selection import train_test_split


current_dir = os.path.dirname(os.path.abspath(__file__)) 
scripts_dir = os.path.dirname(current_dir)               
sys.path.insert(0, scripts_dir)                          

try:
    from config import DATA_DIR
except ImportError:
    DATA_DIR = "./data"


def clean_text(text):
    if pd.isna(text) or str(text).strip() == "":
        return ""
    
    text = str(text)
    text = unicodedata.normalize('NFC', text)
    
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)

    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'(:\)+|=\)+|:\>|:3|:\(|<3|=\(|:\')', '', text)

    text = re.sub(
        r'(\b\d{1,2})[\/\.\-](\d{1,2})[\/\.\-](\d{2,4}\b)',
        lambda m: f"{int(m.group(1))}-{int(m.group(2))}-{int(m.group(3))}",
        text
    )
    text = re.sub(r'(\b\d{1,2})\.(\d{1,2}\b)', r'\1:\2', text)
    
    vietnamese_chars = "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ"
    pattern = r'[^a-zA-Z0-9\s.!?,\-;:()\[\]{}"\'' + vietnamese_chars + r']'
    text = re.sub(pattern, '', text)
    
    text = re.sub(r'[\n\r\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def assign_vietnamese_class(subreddit):
    sub = str(subreddit).lower().strip()
    if 'adhd' in sub:
        return 'rối loạn tăng động giảm chú ý', 0
    elif 'anxiety' in sub:
        return 'rối loạn lo âu', 1
    elif 'bipolar' in sub:
        return 'rối loạn lưỡng cực', 2
    elif 'depression' in sub:
        return 'trầm cảm', 3
    elif 'ptsd' in sub:
        return 'rối loạn căng thẳng sau sang chấn', 4
    else:
        return 'không liên quan đến các bệnh tâm lý', 5


def process_pipeline():
    input_file = os.path.join(DATA_DIR, "original", "dataframe_vi.csv")
    if not os.path.exists(input_file):
        input_file = "dataframe1_vi.csv"
        
    print(f" {input_file} ")
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except FileNotFoundError:
        return

    cols_to_drop = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and (df[col] == 0).all()]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    content_col = 'Content_vi' if 'Content_vi' in df.columns else ('post' if 'post' in df.columns else 'Content')
    
    if 'ID' not in df.columns:
        df.insert(0, 'ID', [str(uuid.uuid4()) for _ in range(len(df))])

    df['title'] = df[title_col].apply(clean_text)
    df['post'] = df[content_col].apply(clean_text)

    df['text'] = df.apply(lambda row: f"{row['title']}: {row['post']}" if row['title'] else row['post'], axis=1)
    
    df = df[df['text'].str.strip() != ''].copy()

    subreddit_col = 'Subreddit(Label of category)'
    if subreddit_col not in df.columns:
        matches = [c for c in df.columns if 'subreddit' in str(c).lower()]
        if matches:
            subreddit_col = matches[0]

    labels = df[subreddit_col].apply(assign_vietnamese_class)
    df['class_name'] = [x[0] for x in labels]
    df['class_id'] = [x[1] for x in labels]

    final_cols = ['ID', 'title', 'post', 'class_name', 'class_id', 'text']
    df_final = df[[col for col in final_cols if col in df.columns]]

    print(df_final['class_name'].value_counts())

    train_df, temp_df = train_test_split(df_final, test_size=0.2, random_state=42, stratify=df_final['class_id'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['class_id'])

    save_dir = os.path.join(DATA_DIR, "processed")
    os.makedirs(save_dir, exist_ok=True)

    train_path = os.path.join(save_dir, "train_vi.csv")
    val_path = os.path.join(save_dir, "val_vi.csv")
    test_path = os.path.join(save_dir, "test_vi.csv")

    train_df.to_csv(train_path, index=False, encoding='utf-8')
    val_df.to_csv(val_path, index=False, encoding='utf-8')
    test_df.to_csv(test_path, index=False, encoding='utf-8')

    print(f"✅ Train (80%): {len(train_df)}  -> {train_path}")
    print(f"✅ Val (10%):   {len(val_df)}  -> {val_path}")
    print(f"✅ Test (10%):  {len(test_df)}  -> {test_path}")

if __name__ == "__main__":
    process_pipeline()