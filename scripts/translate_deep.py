import os
import time
import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm

FOLDER_CHUA_DATA = "your_folder"
FILE_NAME = "dataframe_part_3.xlsx"

OUTPUT_FILE = FILE_NAME.replace(".xlsx", "_vi.csv")
SAVE_INTERVAL = 50

translator = GoogleTranslator(source='en', target='vi')


def translate_with_deep(text, max_retries=3):
    if pd.isna(text) or str(text).strip() == "":
        return text

    text = str(text)
    chunk_size = 4500  

    def _translate_chunk(chunk):
        for attempt in range(max_retries):
            try:
                return translator.translate(chunk)
            except Exception as e:
                tqdm.write(f"  {e}  sleep 5s")
                time.sleep(5)
        tqdm.write(" [!] ")
        return ""

    if len(text) <= chunk_size:
        result = _translate_chunk(text)
        return result if result else None

    translated_chunks = []

    while len(text) > 0:
        if len(text) <= chunk_size:
            translated_chunks.append(_translate_chunk(text))
            break

        cut_idx = max(
            text.rfind('. ', 0, chunk_size),
            text.rfind('\n', 0, chunk_size),
            text.rfind('? ', 0, chunk_size),
            text.rfind('! ', 0, chunk_size)
        )

        if cut_idx == -1:
            cut_idx = chunk_size
        else:
            cut_idx += 1

        chunk = text[:cut_idx].strip()
        translated_chunks.append(_translate_chunk(chunk))

        text = text[cut_idx:].strip()
        time.sleep(1)

    valid_chunks = [c for c in translated_chunks if c]
    if not valid_chunks:
        return None

    return " ".join(valid_chunks)


def safe_save_csv(df, path):
    temp_file = path + ".tmp"
    df.to_csv(temp_file, index=False, encoding="utf-8-sig")
    os.replace(temp_file, path)


input_path = os.path.join(FOLDER_CHUA_DATA, FILE_NAME)
output_path = os.path.join(FOLDER_CHUA_DATA, OUTPUT_FILE)

print(f"\n{'-'*50}\n {FILE_NAME}\n{'-'*50}")

try:
    df = pd.read_excel(input_path, engine="openpyxl")
except FileNotFoundError:
    exit()

if os.path.exists(output_path):
    try:
        df_translated = pd.read_csv(output_path)
    except Exception as e:
        print(f"⚠️ ({e}) ")
        df_translated = df.copy()
else:
    df_translated = df.copy()

columns_to_translate = ["Title", "Content"]

for col in columns_to_translate:
    if col in df_translated.columns:
        col_vi = f"{col}_vi"
        if col_vi not in df_translated.columns:
            df_translated[col_vi] = None

pbar = tqdm(total=len(df_translated), desc="", unit="")

processed_count = 0

for index, row in df_translated.iterrows():
    translated_flag = False

    for col in columns_to_translate:
        col_vi = f"{col}_vi"

        if col in df_translated.columns:
            if pd.isna(row.get(col_vi)) or str(row.get(col_vi)).strip() == "":
                text = row[col]
                result = translate_with_deep(text)

                if result:
                    df_translated.at[index, col_vi] = result
                    translated_flag = True

                time.sleep(1)

    processed_count += 1
    pbar.update(1)

    if translated_flag and processed_count % SAVE_INTERVAL == 0:
        safe_save_csv(df_translated, output_path)

pbar.close()

safe_save_csv(df_translated, output_path)
print(f"📁 File output: {os.path.abspath(output_path)}")