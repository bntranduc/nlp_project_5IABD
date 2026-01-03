import pandas as pd
from datasets import load_dataset


def make_dataset(file_path):
    dataset = load_dataset('json', data_files=file_path, split='train')
    df = dataset.to_pandas()
    X = df["raw_log"]
    y = df["severity"]
    return X, y


if __name__ == "__main__":
    X, y = make_dataset("/home/bao/nlp_project_5IABD/raw/SIEM_Dataset/advanced_siem_dataset.jsonl")
    print(X.head())
    print(y.head())