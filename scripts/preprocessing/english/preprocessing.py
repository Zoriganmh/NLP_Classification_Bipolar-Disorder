import os
import re
import uuid
import pandas as pd
import emoji
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

DATA_DIR = "./data" 
INPUT_FILE = "dataframe.xlsx" 
OUTPUT_PREFIX = "english"


def clean_text_english(text):
    if pd.isna(text) or str(text).strip() == "":
        return ""

    text = str(text)

    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = BeautifulSoup(text, "html.parser").get_text()

    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'(:\)+|=\)+|:\>|:3|:\(|<3|=\(|:\')', '', text)

    text = re.sub(
        r'(\b\d{1,2})[\/\.\-](\d{1,2})[\/\.\-](\d{2,4}\b)',
        lambda m: f"{int(m.group(1))}-{int(m.group(2))}-{int(m.group(3))}",
        text
    )
    text = re.sub(r'(\b\d{1,2})\.(\d{1,2}\b)', r'\1:\2', text)

    allowed = r"[^0-9A-Za-z\s\.\,\?\!\:\;\'\"\(\)\[\]\{\}\-–]"
    text = re.sub(allowed, '', text)
    text = re.sub(r'(?<![A-Za-z])-(?![A-Za-z])', '', text)

    text = re.sub(r'([?!])\1+', r'\1', text)
    text = re.sub(r'^[\?\!\.\,;:]+(?=\w)', '', text)
    text = re.sub(r'([.,;:])\s*', r'\1 ', text)
    text = re.sub(r'\(\s+', '(', text)   
    text = re.sub(r'\s+\)', ')', text)   
    text = re.sub(r'(\d+\))\s*', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def apply_label_logic(text):
    sub = str(text).lower()
    if 'adhd' in sub:
        return 'adhd', 0
    elif 'anxiety' in sub:
        return 'anxiety', 1
    elif 'bipolar' in sub:
        return 'bipolar', 2
    elif 'depression' in sub:
        return 'depression', 3
    elif 'ptsd' in sub:
        return 'ptsd', 4
    else:
        return 'none', 5


def process_pipeline():
    print(f"{INPUT_FILE} ---")
    df = pd.read_excel(INPUT_FILE)

    cols_to_drop = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and (df[col] == 0).all()]
    if cols_to_drop:
        print(f"{cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    rename_map = {}
    if 'Title' in df.columns: rename_map['Title'] = 'title'
    if 'Content' in df.columns: rename_map['Content'] = 'post'
    df = df.rename(columns=rename_map)
    
    if 'title' not in df.columns: df['title'] = ''
    if 'post' not in df.columns: df['post'] = ''

    if 'ID' not in df.columns:
        df.insert(0, 'ID', [str(uuid.uuid4()) for _ in range(len(df))])

    df['title'] = df['title'].apply(clean_text_english)
    df['post'] = df['post'].apply(clean_text_english)

    df = df[(df['title'] != '') | (df['post'] != '')].copy()

    text_to_scan = df['title'].fillna('') + " " + df['post'].fillna('')
    labels = text_to_scan.apply(apply_label_logic)

    df['class_name'] = [x[0] for x in labels]
    df['class_id'] = [x[1] for x in labels]

    df['title_str'] = df['title'].fillna('')
    df['post_str'] = df['post'].fillna('')
    df['text'] = df.apply(lambda row: f"{row['title_str']}: {row['post_str']}" if row['title_str'] else row['post_str'], axis=1)

    final_cols = ['ID', 'title', 'post', 'class_name', 'class_id', 'text']
    df = df[[col for col in final_cols if col in df.columns]]

    print(df['class_name'].value_counts())

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class_id'])
    
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['class_id'])

    save_dir = os.path.join(DATA_DIR, "processed")
    os.makedirs(save_dir, exist_ok=True)

    train_path = os.path.join(save_dir, f"{OUTPUT_PREFIX}_train.csv")
    val_path = os.path.join(save_dir, f"{OUTPUT_PREFIX}_val.csv")
    test_path = os.path.join(save_dir, f"{OUTPUT_PREFIX}_test.csv")

    train_df.to_csv(train_path, index=False, encoding='utf-8')
    val_df.to_csv(val_path, index=False, encoding='utf-8')
    test_df.to_csv(test_path, index=False, encoding='utf-8')

    print(f" {train_path} ({len(train_df)} ")
    print(f"   {val_path} ({len(val_df)} ")
    print(f"  {test_path} ({len(test_df)} ")

if __name__ == "__main__":
    process_pipeline()