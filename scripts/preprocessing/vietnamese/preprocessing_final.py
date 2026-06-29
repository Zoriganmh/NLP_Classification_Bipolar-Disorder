import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)
sys.path.insert(0, scripts_dir)

try:
    from config import DATA_DIR
except ImportError:
    DATA_DIR = "./data"


def process_pipeline():

    input_file = "Final_Merged_Dataset.csv"

    try:
        df = pd.read_csv(input_file, encoding="utf-8-sig")

    except FileNotFoundError:
        return

    except Exception as e:
        return

    print(f"{len(df):,}")

    required_columns = [
        "ID",
        "title",
        "post",
        "class_name",
        "class_id",
        "text"
    ]

    missing_cols = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_cols:
        return

    label_stats = (
        df["class_name"]
        .value_counts()
        .to_frame(name="count")
    )

    label_stats["percentage"] = (
        label_stats["count"]
        / len(df)
        * 100
    ).round(2)

    print(label_stats)

    print(
        df[
            [
                "title",
                "post",
                "text",
                "class_name",
                "class_id"
            ]
        ]
        .isnull()
        .sum()
    )

    before_rows = len(df)

    df = df[
        df["text"]
        .fillna("")
        .str.strip()
        != ""
    ].copy()

    after_rows = len(df)

    print(f"Removed {before_rows - after_rows:,} empty rows.")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.10,
        random_state=42,
        stratify=df["class_id"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["class_id"]
    )

    print(f"Train      : {len(train_df):,}")
    print(f"Validation : {len(val_df):,}")
    print(f"Test       : {len(test_df):,}")

    print(
        f"{len(train_df) + len(val_df) + len(test_df):,}"
    )

    print(
        f"\nTrain      : {len(train_df)/len(df)*100:.2f}%"
    )

    print(
        f"Validation : {len(val_df)/len(df)*100:.2f}%"
    )

    print(
        f"Test       : {len(test_df)/len(df)*100:.2f}%"
    )

    save_dir = os.path.join(
        DATA_DIR,
        "processed"
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    train_path = os.path.join(
        save_dir,
        "train_vi.csv"
    )

    val_path = os.path.join(
        save_dir,
        "val_vi.csv"
    )

    test_path = os.path.join(
        save_dir,
        "test_vi.csv"
    )

    train_df.to_csv(
        train_path,
        index=False,
        encoding="utf-8-sig"
    )

    val_df.to_csv(
        val_path,
        index=False,
        encoding="utf-8-sig"
    )

    test_df.to_csv(
        test_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"✅ Train ({len(train_df):,}) -> "
        f"{train_path}"
    )

    print(
        f"✅ Validation ({len(val_df):,}) -> "
        f"{val_path}"
    )

    print(
        f"✅ Test ({len(test_df):,}) -> "
        f"{test_path}"
    )

    train_stats = (
        train_df["class_name"]
        .value_counts()
        .to_frame(name="count")
    )

    train_stats["percentage"] = (
        train_stats["count"]
        / len(train_df)
        * 100
    ).round(2)

    print(train_stats)

    val_stats = (
        val_df["class_name"]
        .value_counts()
        .to_frame(name="count")
    )

    val_stats["percentage"] = (
        val_stats["count"]
        / len(val_df)
        * 100
    ).round(2)

    print(val_stats)

    test_stats = (
        test_df["class_name"]
        .value_counts()
        .to_frame(name="count")
    )

    test_stats["percentage"] = (
        test_stats["count"]
        / len(test_df)
        * 100
    ).round(2)

    print(test_stats)


if __name__ == "__main__":
    process_pipeline()