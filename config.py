import os

current_dir = os.path.dirname(os.path.abspath(__file__)) 
BASE_DIR = os.path.dirname(current_dir)                  


DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')

# --- File Names & Paths ---
TRAIN_FILE = "train.csv"
VAL_FILE = "val.csv"
TEST_FILE = "test.csv"

TRAIN_PATH = os.path.join(DATA_DIR, "processed", TRAIN_FILE)
VAL_PATH = os.path.join(DATA_DIR, "processed", VAL_FILE)
TEST_PATH = os.path.join(DATA_DIR, "processed", TEST_FILE)

# --- Model Paths (english) ---
ROBERTA_MODEL_NAME = "roberta-base"
ROBERTA_MODEL_DIR = os.path.join(MODEL_DIR, "roberta")
EMOTION_MODEL_NAME = "SamLowe/roberta-base-go_emotions"

# --- Model Paths (vietnamese) ---
PHOBERT_MODEL_NAME = "vinai/phobert-base-v2"
PHOBERT_MODEL_DIR = os.path.join(MODEL_DIR, "phobert")

# --- Results Paths ---
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")
TOKENIZED_DIR = os.path.join(FEATURES_DIR, "roberta")
EVAL_DIR = os.path.join(OUTPUT_DIR, "evaluation")
STAT_DIR = os.path.join(OUTPUT_DIR, "statistic")
LOGGING_DIR = os.path.join(OUTPUT_DIR, "logs")

# --- Training Configuration ---
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

# --- Class Labels and Other Metadata ---
LABEL_COL = "class_id"
TEXT_COL = "text"
NUM_LABELS = 6

# --- Logging and Saving Results ---
SAVE_MODEL = True
SAVE_RESULTS = True