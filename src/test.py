"""
Transfer learning script for SIEM dataset severity classification.
Inspired by TD2_transfer_learning.ipynb, adapted for sequence classification.
"""

import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
import evaluate


# Configuration
MODEL_NAME = "distilbert/distilbert-base-cased"
TEXT_FIELD = "raw_log"  # or "description" if preferred
LABEL_FIELD = "severity"


def load_siem_dataset(file_path):
    """
    Load SIEM dataset from JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the data
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def prepare_data(data, text_field=TEXT_FIELD, label_field=LABEL_FIELD, max_samples=None):
    """
    Prepare data for training: extract text and labels.
    
    Args:
        data: List of dictionaries from the dataset
        text_field: Field name containing the text to classify
        label_field: Field name containing the labels
        max_samples: Maximum number of samples to use (for testing)
        
    Returns:
        sentences: List of text strings
        labels: List of label strings
    """
    sentences = []
    labels = []
    
    data_subset = data[:max_samples] if max_samples else data
    
    for item in data_subset:
        text = item.get(text_field, "")
        label = item.get(label_field, "")
        
        if text and label:  # Only include non-empty entries
            sentences.append(text)
            labels.append(label)
    
    return sentences, labels


def tokenize_data(sentences, tokenizer, max_length=512):
    """
    Tokenize sentences using the tokenizer.
    
    Args:
        sentences: List of text strings
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Tokenized inputs dictionary
    """
    return tokenizer(
        sentences,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt" if torch.cuda.is_available() else None,
    )


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
    
    Args:
        eval_pred: Evaluation predictions tuple (predictions, labels)
        
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    
    results = {}
    results.update(accuracy_metric.compute(predictions=predictions, references=labels))
    results.update(f1_metric.compute(predictions=predictions, references=labels, average="weighted"))
    results.update(precision_metric.compute(predictions=predictions, references=labels, average="weighted"))
    results.update(recall_metric.compute(predictions=predictions, references=labels, average="weighted"))
    
    return results


def freeze_model_layers(model, unfreeze_last_n_layers=1):
    """
    Freeze all layers except the last N transformer layers.
    Inspired by the notebook approach - freezes base model and only unfreezes
    the last layer's FFN components.
    
    Args:
        model: The transformer model
        unfreeze_last_n_layers: Number of last layers to unfreeze (default: 1)
    """
    # Freeze all base model parameters first
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    # Unfreeze the last N layers (similar to notebook approach)
    # The notebook unfreezes layer.5's FFN components
    if hasattr(model.base_model, "transformer"):
        # For DistilBERT
        layers = model.base_model.transformer.layer
        num_layers = len(layers)
        
        # Unfreeze the last N layers' FFN components (like in the notebook)
        for i in range(max(0, num_layers - unfreeze_last_n_layers), num_layers):
            for name, param in layers[i].named_parameters():
                # Unfreeze FFN layers (like in notebook: "ffn.lin" in name)
                if "ffn.lin" in name:
                    param.requires_grad = True
    
    # Always unfreeze the classifier head
    for param in model.classifier.parameters():
        param.requires_grad = True


def main():
    """Main training function."""
    
    # Load data
    print("Loading dataset...")
    file_path = "data/raw/SIEM_Dataset/advanced_siem_dataset.jsonl"
    data = load_siem_dataset(file_path)
    print(f"Loaded {len(data)} samples")
    
    # Prepare data (using subset for faster iteration - remove max_samples for full dataset)
    print("Preparing data...")
    sentences, labels = prepare_data(data, max_samples=10000)  # Remove max_samples for full dataset
    print(f"Prepared {len(sentences)} samples")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_labels = len(label_encoder.classes_)
    print(f"Number of classes: {num_labels}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Split data: train -> (train, dev), then separate test
    print("Splitting data...")
    sentences_training, sentences_test, labels_training, labels_test = train_test_split(
        sentences,
        encoded_labels,
        test_size=0.2,
        random_state=42,
        stratify=encoded_labels,
    )
    
    sentences_train, sentences_dev, labels_train, labels_dev = train_test_split(
        sentences_training,
        labels_training,
        test_size=0.2,
        random_state=42,
        stratify=labels_training,
    )
    
    print(f"Train: {len(sentences_train)}, Dev: {len(sentences_dev)}, Test: {len(sentences_test)}")
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Tokenize data
    print("Tokenizing data...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)
    
    # Create datasets
    train_dict = {"text": sentences_train, "labels": labels_train.tolist()}
    dev_dict = {"text": sentences_dev, "labels": labels_dev.tolist()}
    test_dict = {"text": sentences_test, "labels": labels_test.tolist()}
    
    dataset_train = Dataset.from_dict(train_dict)
    dataset_dev = Dataset.from_dict(dev_dict)
    dataset_test = Dataset.from_dict(test_dict)
    
    dataset_train = dataset_train.map(tokenize_function, batched=True)
    dataset_dev = dataset_dev.map(tokenize_function, batched=True)
    dataset_test = dataset_test.map(tokenize_function, batched=True)
    
    # Initialize model
    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )
    
    # Freeze layers (only train last layer + classifier)
    print("Freezing model layers...")
    freeze_model_layers(model, unfreeze_last_n_layers=1)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models/siem_severity_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_dir="./logs",
        logging_steps=100,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(dataset_test)
    print(f"Test results: {test_results}")
    
    # Save model and label encoder
    print("Saving model...")
    trainer.save_model("./models/siem_severity_model_final")
    tokenizer.save_pretrained("./models/siem_severity_model_final")
    
    # Save label encoder (using pickle or json)
    import pickle
    with open("./models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
