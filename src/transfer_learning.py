"""
Transfer learning script for SIEM dataset severity classification.
Inspired by TD2_transfer_learning.ipynb, adapted for sequence classification.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset


# Configuration
MODEL_NAME = "distilbert/distilbert-base-cased"
TEXT_FIELD = "description"
LABEL_FIELD = "severity"


def load_data_from_csv(train_csv_path, test_csv_path, text_field=TEXT_FIELD, label_field=LABEL_FIELD):
    """
    Charge les données depuis les fichiers CSV.
    
    Args:
        train_csv_path: Chemin vers train.csv
        test_csv_path: Chemin vers test.csv
        text_field: Nom du champ texte
        label_field: Nom du champ label
        
    Returns:
        sentences_train_full, labels_train_full, sentences_test, labels_test
    """
    print("Loading dataset from CSV files...")
    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)
    
    print(f"Train: {len(df_train)} samples")
    print(f"Test: {len(df_test)} samples")
    
    # Extract text and labels
    sentences_train_full = df_train[text_field].fillna("").astype(str).tolist()
    labels_train_full = df_train[label_field].tolist()
    
    sentences_test = df_test[text_field].fillna("").astype(str).tolist()
    labels_test = df_test[label_field].tolist()
    
    return sentences_train_full, labels_train_full, sentences_test, labels_test


def split_train_dev(sentences_train_full, labels_train_full, test_size=0.2, random_state=42):
    """
    Divise le train en train et dev.
    
    Args:
        sentences_train_full: Liste des phrases d'entraînement
        labels_train_full: Liste des labels d'entraînement
        test_size: Proportion pour dev
        random_state: Seed
        
    Returns:
        sentences_train, sentences_dev, labels_train, labels_dev
    """
    print("Splitting train into train and dev...")
    sentences_train, sentences_dev, labels_train, labels_dev = train_test_split(
        sentences_train_full,
        labels_train_full,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_train_full,
    )
    
    print(f"Train: {len(sentences_train)}, Dev: {len(sentences_dev)}")
    
    return sentences_train, sentences_dev, labels_train, labels_dev


def encode_labels(labels_train, labels_dev, labels_test):
    """
    Encode les labels avec LabelEncoder.
    
    Args:
        labels_train: Labels d'entraînement
        labels_dev: Labels de dev
        labels_test: Labels de test
        
    Returns:
        label_encoder, labels_train_encoded, labels_dev_encoded, labels_test_encoded, num_labels
    """
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    
    # Fit on all labels to ensure consistency
    all_labels = labels_train + labels_dev + labels_test
    label_encoder.fit(all_labels)
    
    labels_train_encoded = label_encoder.transform(labels_train)
    labels_dev_encoded = label_encoder.transform(labels_dev)
    labels_test_encoded = label_encoder.transform(labels_test)
    
    num_labels = len(label_encoder.classes_)
    print(f"Number of classes: {num_labels}")
    print(f"Classes: {label_encoder.classes_}")
    
    return label_encoder, labels_train_encoded, labels_dev_encoded, labels_test_encoded, num_labels


def create_tokenized_datasets(sentences_train, sentences_dev, sentences_test,
                              labels_train_encoded, labels_dev_encoded, labels_test_encoded,
                              model_name=MODEL_NAME, max_length=512):
    """
    Crée les datasets tokenisés pour HuggingFace.
    
    Args:
        sentences_train: Phrases d'entraînement
        sentences_dev: Phrases de dev
        sentences_test: Phrases de test
        labels_train_encoded: Labels encodés d'entraînement
        labels_dev_encoded: Labels encodés de dev
        labels_test_encoded: Labels encodés de test
        model_name: Nom du modèle
        max_length: Longueur maximale des séquences
        
    Returns:
        tokenizer, dataset_train, dataset_dev, dataset_test
    """
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Tokenizing data...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)
    
    # Create datasets
    train_dict = {"text": sentences_train, "labels": labels_train_encoded.tolist()}
    dev_dict = {"text": sentences_dev, "labels": labels_dev_encoded.tolist()}
    test_dict = {"text": sentences_test, "labels": labels_test_encoded.tolist()}
    
    dataset_train = Dataset.from_dict(train_dict)
    dataset_dev = Dataset.from_dict(dev_dict)
    dataset_test = Dataset.from_dict(test_dict)
    
    dataset_train = dataset_train.map(tokenize_function, batched=True)
    dataset_dev = dataset_dev.map(tokenize_function, batched=True)
    dataset_test = dataset_test.map(tokenize_function, batched=True)
    
    return tokenizer, dataset_train, dataset_dev, dataset_test


def create_model(model_name=MODEL_NAME, num_labels=6, unfreeze_last_n_layers=1):
    """
    Crée et configure le modèle avec freeze des couches.
    
    Args:
        model_name: Nom du modèle pré-entraîné
        num_labels: Nombre de classes
        unfreeze_last_n_layers: Nombre de dernières couches à dégeler
        
    Returns:
        model
    """
    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    
    # Freeze layers (only train last layer + classifier)
    print("Freezing model layers...")
    freeze_model_layers(model, unfreeze_last_n_layers=unfreeze_last_n_layers)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model


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
    
    # Calculate metrics using sklearn
    accuracy = accuracy_score(labels, predictions)
    f1_weighted = f1_score(labels, predictions, average="weighted")
    f1_macro = f1_score(labels, predictions, average="macro")
    precision = precision_score(labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(labels, predictions, average="weighted", zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "precision": precision,
        "recall": recall,
    }

from torch.utils.data import WeightedRandomSampler, DataLoader

def create_stratified_sampler(labels_train_encoded):
    """
    Crée un sampler stratifié pour équilibrer les batches pendant l'entraînement.
    
    Args:
        labels_train_encoded: Labels encodés du train set
        
    Returns:
        WeightedRandomSampler
    """
    from collections import Counter
    
    # Compter les occurrences de chaque classe
    class_counts = Counter(labels_train_encoded)
    print(f"\nClass distribution in training set:")
    for label, count in sorted(class_counts.items()):
        print(f"  Class {label}: {count} samples ({100*count/len(labels_train_encoded):.2f}%)")
    
    # Calculer les poids pour chaque échantillon
    # weight = 1 / count_of_class
    class_weights = {label: 1.0 / count for label, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels_train_encoded]
    
    # Créer le sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def create_trainer(model, tokenizer, dataset_train, dataset_dev, 
                   labels_train_encoded,
                   output_dir="./models/siem_severity_model",
                   learning_rate=3e-5,
                   per_device_train_batch_size=32,
                   per_device_eval_batch_size=64,
                   num_train_epochs=15,
                   weight_decay=0.01,
                   logging_dir="./logs",
                   logging_steps=100,
                   warmup_ratio=0.1,
                   use_focal_loss=False,
                   focal_gamma=2.0,
                   weight_scaling=0.3,
                   use_stratified_sampling=False):
    """
    Crée le Trainer avec class weights réduits, focal loss, et stratified sampling.
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Calculer les class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels_train_encoded),
        y=labels_train_encoded
    )
    
    # Réduire drastiquement les poids
    class_weights_reduced = 1 + (class_weights - 1) * weight_scaling
    class_weights_tensor = torch.FloatTensor(class_weights_reduced)
    
    print(f"\nClass weights (original): {class_weights}")
    print(f"Class weights (reduced with scaling={weight_scaling}): {class_weights_reduced}")
    
    # Focal Loss
    class FocalLoss(torch.nn.Module):
        def __init__(self, alpha=None, gamma=2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
        
        def forward(self, inputs, targets):
            ce_loss = torch.nn.functional.cross_entropy(
                inputs, targets, reduction='none', weight=self.alpha
            )
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
            return focal_loss
    
    # STRATÉGIE 3: Stratified Sampling
    train_sampler = None
    if use_stratified_sampling:
        print("\n🎯 Using stratified sampling for balanced batches")
        train_sampler = create_stratified_sampler(labels_train_encoded)
    
    # Custom Trainer avec support du sampler
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            if use_focal_loss:
                loss_fct = FocalLoss(
                    alpha=class_weights_tensor.to(model.device),
                    gamma=focal_gamma
                )
                loss = loss_fct(logits, labels)
            else:
                loss_fct = torch.nn.CrossEntropyLoss(
                    weight=class_weights_tensor.to(model.device)
                )
                loss = loss_fct(logits, labels)
            
            return (loss, outputs) if return_outputs else loss
        
        def _get_train_sampler(self, train_dataset):  # CORRECTION ICI: accepter train_dataset
            """Override pour utiliser notre sampler stratifié."""
            if train_sampler is not None:
                return train_sampler
            return super()._get_train_sampler(train_dataset)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        push_to_hub=False,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
    )
    
    # Initialize WeightedTrainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    return trainer


def train_model(trainer):
    """
    Entraîne le modèle.
    
    Args:
        trainer: Trainer HuggingFace
        
    Returns:
        trainer (après entraînement)
    """
    print("Starting training...")
    trainer.train()
    return trainer


def evaluate_and_save_predictions(trainer, dataset_test, label_encoder, labels_test,
                                   predictions_dir="./predictions",
                                   predictions_filename="transfer_learning_predictions.csv"):
    """
    Évalue le modèle et sauvegarde les prédictions.
    
    Args:
        trainer: Trainer entraîné
        dataset_test: Dataset de test
        label_encoder: LabelEncoder utilisé
        labels_test: Labels originaux de test
        predictions_dir: Répertoire pour sauvegarder les prédictions
        predictions_filename: Nom du fichier de prédictions
        
    Returns:
        predictions_df: DataFrame avec les prédictions
    """
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(dataset_test)
    print(f"Test results: {test_results}")
    
    # Faire les prédictions sur le test set
    print("Making predictions on test set...")
    predictions = trainer.predict(dataset_test)
    y_pred_proba = predictions.predictions
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Convertir les labels prédits en labels originaux
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    # Utiliser les labels originaux (pas encodés) pour sauvegarder
    y_test_labels = labels_test
    
    # Sauvegarder les prédictions dans un CSV
    os.makedirs(predictions_dir, exist_ok=True)
    
    predictions_df = pd.DataFrame({
        "y_true": y_test_labels,
        "y_pred": y_pred_labels
    })
    
    predictions_path = f"{predictions_dir}/{predictions_filename}"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\n✅ Prédictions sauvegardées dans {predictions_path}")
    print(f"   Échantillons: {len(predictions_df)}")
    
    return predictions_df


def save_model(trainer, tokenizer, label_encoder, model_dir="./models/siem_severity_model_final"):
    """
    Sauvegarde le modèle, le tokenizer et le label encoder.
    
    Args:
        trainer: Trainer entraîné
        tokenizer: Tokenizer
        label_encoder: LabelEncoder
        model_dir: Répertoire pour sauvegarder le modèle
    """
    print("Saving model...")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    # Save label encoder
    label_encoder_path = f"{model_dir}/label_encoder.pkl"
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    
    print(f"✅ Modèle sauvegardé dans {model_dir}")


def main():
    """Main training function."""
    
    # Configuration
    train_csv_path = "./data/raw/SIEM_Dataset/train.csv"
    test_csv_path = "./data/raw/SIEM_Dataset/test.csv"

    print(f"Device : {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    # 1. Load data
    sentences_train_full, labels_train_full, sentences_test, labels_test = load_data_from_csv(
        train_csv_path, test_csv_path, TEXT_FIELD, LABEL_FIELD
    )
    
    # 2. Split train into train and dev
    sentences_train, sentences_dev, labels_train, labels_dev = split_train_dev(
        sentences_train_full, labels_train_full
    )
    
    print(f"Test: {len(sentences_test)}")
    
    # 3. Encode labels
    label_encoder, labels_train_encoded, labels_dev_encoded, labels_test_encoded, num_labels = encode_labels(
        labels_train, labels_dev, labels_test
    )
    
    # 4. Create tokenized datasets
    tokenizer, dataset_train, dataset_dev, dataset_test = create_tokenized_datasets(
        sentences_train, sentences_dev, sentences_test,
        labels_train_encoded, labels_dev_encoded, labels_test_encoded
    )
    
    # 5. Create model with more unfrozen layers
    model = create_model(MODEL_NAME, num_labels, unfreeze_last_n_layers=2)
    
    # 6. Create trainer with class weights
    trainer = create_trainer(
        model, tokenizer, dataset_train, dataset_dev,
        labels_train_encoded=labels_train_encoded,
        learning_rate=3e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=15,
        warmup_ratio=0.1,
        use_focal_loss=True,
        focal_gamma=1.5,  # Gamma modéré (essayez 1.0, 1.5, 2.0)
        weight_scaling=0.3,
    )
    
    # 7. Train model
    trainer = train_model(trainer)
    
    # 8. Evaluate and save predictions
    predictions_df = evaluate_and_save_predictions(
        trainer, dataset_test, label_encoder, labels_test
    )
    
    # 9. Save model
    save_model(trainer, tokenizer, label_encoder)
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
