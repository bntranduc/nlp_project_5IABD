from datasets import load_dataset

dataset = load_dataset("bn-tran-duc/Advanced_SIEM_Dataset", split='train')
print(dataset)