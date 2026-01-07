import pandas as pd
from sklearn.model_selection import train_test_split

# -------- CONFIG --------
INPUT_JSON = "data/raw/SIEM_Dataset/advanced_siem_dataset.jsonl"
OUTPUT_DIR = "data/raw/SIEM_Dataset"
TEST_SIZE = 0.2
RANDOM_STATE = 42
# ------------------------


df = pd.read_json(INPUT_JSON,
    lines=True,
    encoding="utf-8",
)


df = df[["description",'raw_log' "severity"]].dropna()


train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["severity"],
    random_state=RANDOM_STATE
)


train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)