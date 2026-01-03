from datasets import load_dataset
from pathlib import Path
import pandas as pd

def load_siem_dataset_hf(file_path=None):    
    file_path = "/home/bao/nlp_project_5IABD/raw/SIEM_Dataset/advanced_siem_dataset.jsonl"
    
    dataset = load_dataset('json', data_files=file_path, split='train')
    # Convertir en DataFrame pandas
    df = dataset.to_pandas()
    return df

df = load_siem_dataset_hf()
print(f"Dataset chargé: {len(df)} lignes, {len(df.columns)} colonnes")
print(f"\nPremières lignes:")
print(df.head())