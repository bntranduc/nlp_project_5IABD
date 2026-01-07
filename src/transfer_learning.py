"""
Transfer learning script for SIEM dataset severity classification.
Inspired by TD2_transfer_learning.ipynb, adapted for sequence classification.
"""

import os
import pickle
import re
import numpy as np
import pandas as pd
import torch
from collections import Counter
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


# ============================================================================
# CONFIGURATION 
# ============================================================================

# Paths
TRAIN_CSV_PATH = "./data/raw/SIEM_Dataset/train.csv"
TEST_CSV_PATH = "./data/raw/SIEM_Dataset/test.csv"
TEXT_FIELD = "description"
LABEL_FIELD = "severity"

# Label mapping (None = pas de fusion, dict = fusion des labels)
# Par défaut: fusion critical + emergency → critical_emergency (5 classes)
LABEL_MAPPING = {"critical": "critical_emergency", "emergency": "critical_emergency"}
MERGED_LABEL = "critical_emergency"

# Pré-traitement des logs
ENABLE_PREPROCESSING = True  # Activer/désactiver le pré-traitement
PREPROCESSING_CONFIG = {
    "lowercase": True,  # Convertir en minuscules
    "remove_urls": True,  # Supprimer les URLs (remplacées par <URL>)
    "remove_ips": True,  # Supprimer les adresses IP (remplacées par <IP>)
    "remove_emails": True,  # Supprimer les emails (remplacés par <EMAIL>)
    "remove_usernames": True,  # Supprimer les noms d'utilisateurs (remplacés par <USER>)
    "normalize_ports": False,  # Normaliser les ports (remplacés par <PORT>)
    "normalize_http_codes": False,  # Normaliser les codes HTTP (remplacés par <HTTP_CODE>)
    "normalize_event_ids": False,  # Normaliser les event IDs (remplacés par <EVENT_ID>)
    "normalize_sizes": False,  # Normaliser les tailles/volumes (remplacés par <SIZE>)
    "normalize_timestamps": False,  # Normaliser les timestamps (remplacés par <TS>)
    "normalize_paths": False,  # Normaliser les chemins de fichiers (remplacés par <PATH>)
    "normalize_hashes": False,  # Normaliser les hashes (remplacés par <HASH>)
    "normalize_guids": False,  # Normaliser les GUID/UUID (remplacés par <GUID>)
    "normalize_mac_addresses": False,  # Normaliser les adresses MAC (remplacées par <MAC>)
    "normalize_hostnames": False,  # Normaliser les noms de machines (remplacés par <HOSTNAME>)
    "normalize_threat_actors": False,  # Normaliser les Associated Threat Actor (remplacés par <THREAT_ACTOR>)
    "remove_extra_spaces": True,  # Normaliser les espaces multiples
    "remove_special_chars": False,  # Garder les caractères spéciaux par défaut
}

# Model configuration
MODEL_NAME = 'cisco-ai/SecureBERT2.0-base'
UNFREEZE_LAST_N_LAYERS = 2
MAX_LENGTH = 512

# Training configuration
LEARNING_RATE = 3e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 64
PER_DEVICE_EVAL_BATCH_SIZE = 64
NUM_TRAIN_EPOCHS = 15
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
LR_SCHEDULER_TYPE = "cosine"

# Loss configuration
USE_FOCAL_LOSS = False
FOCAL_GAMMA = 1.5
WEIGHT_SCALING = 0.3  # Scaling des class weights (0.0 = pas de weights, 1.0 = weights complets)
USE_STRATIFIED_SAMPLING = False

# Output configuration
OUTPUT_DIR = "./models/siem_severity_model"
PREDICTIONS_DIR = "./predictions"
PREDICTIONS_FILENAME = "transfer_learning_predictions.csv"
MODEL_DIR = "./models/siem_severity_model_final"
LOGGING_DIR = "./logs"
LOGGING_STEPS = 100

# Data split configuration
DEV_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# FONCTIONS DE PRÉ-TRAITEMENT
# ============================================================================


def normalize_label(label: object) -> str:
    """Normalise un label."""
    return str(label).strip().lower()


def preprocess_text(text: str, config: dict) -> str:
    """
    Prétraite un texte selon la configuration.
    
    Args:
        text: Texte à prétraiter
        config: Configuration du pré-traitement
        
    Returns:
        Texte prétraité
    """
    if not text or not isinstance(text, str):
        return ""
    
    result = text
    
    # Lowercase
    if config.get("lowercase", False):
        result = result.lower()
    
    # Remove URLs
    if config.get("remove_urls", False):
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        result = re.sub(url_pattern, '<URL>', result)
    
    # Normalize ports (AVANT suppression IPs pour capturer IP:PORT)
    if config.get("normalize_ports", False):
        # Pattern: IP:PORT (remplace le port par <PORT> mais garde l'IP pour traitement ultérieur)
        ip_port_pattern = r'\b((?:\d{1,3}\.){3}\d{1,3}):(\d{1,5})\b'
        result = re.sub(ip_port_pattern, r'\1:<PORT>', result)
        # Pattern: :PORT seul (déjà séparé de l'IP)
        port_pattern = r':(\d{1,5})\b'
        result = re.sub(port_pattern, ':<PORT>', result)
    
    # Remove IPs
    if config.get("remove_ips", False):
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        result = re.sub(ip_pattern, '<IP>', result)
    
    # Remove emails
    if config.get("remove_emails", False):
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        result = re.sub(email_pattern, '<EMAIL>', result)
    
    # Remove usernames
    if config.get("remove_usernames", False):
        # Pattern 1: Format Windows/AD: DOMAIN\username ou DOMAIN/username
        # Ex: CORP\john.doe, MYDOMAIN/admin
        domain_user_pattern = r'\b[A-Za-z0-9_-]+[\\/][A-Za-z0-9._-]+\b'
        result = re.sub(domain_user_pattern, '<USER>', result)
        
        # Pattern 2: Format Unix: username@hostname (après suppression emails)
        # On cherche username@hostname où hostname n'a pas de TLD (pas de .com, .org, etc.)
        # Ex: root@server01, admin@localhost, user@hostname
        # Les emails complets ont déjà été supprimés, donc ce qui reste sont des usernames Unix
        unix_user_pattern = r'\b[A-Za-z0-9._-]+@[A-Za-z0-9._-]+\b'
        result = re.sub(unix_user_pattern, '<USER>', result)
        
        # Pattern 3: Usernames dans les logs SIEM après "by" ou "for"
        # Ex: "by amanda15", "by kingjames", "for paulhernandez", "for grayalexander"
        # Format: lettres + chiffres optionnels (amanda15, matthew51) ou juste lettres (kingjames, cgamble)
        # Username pattern: au moins 3 caractères, lettres + chiffres optionnels, peut contenir points/underscores
        siem_user_pattern = r'\b(by|for)\s+[A-Za-z][A-Za-z0-9._-]{2,}\b'
        result = re.sub(siem_user_pattern, r'\1 <USER>', result, flags=re.IGNORECASE)
        
        # Pattern 4: Patterns avec mots-clés spécifiques dans les logs SIEM
        # user:username, username:value, account:username, login:username
        user_keywords = ['user', 'username', 'account', 'login', 'uid', 'principal']
        for keyword in user_keywords:
            # Format: keyword:username ou keyword=username
            pattern = rf'\b{keyword}\s*[:=]\s*[A-Za-z0-9._-]+\b'
            result = re.sub(pattern, f'{keyword}:<USER>', result, flags=re.IGNORECASE)
    
    # Normalize HTTP codes (3-digit codes: 200, 404, 500, etc.)
    if config.get("normalize_http_codes", False):
        # Pattern: codes HTTP (100-599) souvent précédés de "HTTP", "status", "code"
        http_code_pattern = r'\b(?:HTTP|http|status|code)\s*[:=]?\s*([1-5]\d{2})\b'
        result = re.sub(http_code_pattern, r'<HTTP_CODE>', result, flags=re.IGNORECASE)
        # Aussi capturer les codes seuls dans certains contextes
        http_code_standalone = r'\b([1-5]\d{2})\s+(?:OK|Not Found|Forbidden|Unauthorized|Internal Server Error|Bad Request)\b'
        result = re.sub(http_code_standalone, r'<HTTP_CODE>', result, flags=re.IGNORECASE)
    
    # Normalize event IDs (longs nombres souvent avec "event_id", "id:", etc.)
    if config.get("normalize_event_ids", False):
        # Pattern: event_id:12345, id:123456, event 1234567
        event_id_pattern = r'\b(?:event[_-]?id|id|event)\s*[:=]?\s*(\d{4,})\b'
        result = re.sub(event_id_pattern, r'<EVENT_ID>', result, flags=re.IGNORECASE)
        # Aussi capturer les très longs nombres seuls (probablement des IDs)
        long_number_pattern = r'\b\d{8,}\b'
        result = re.sub(long_number_pattern, '<EVENT_ID>', result)
    
    # Normalize sizes/volumes (nombres avec unités: KB, MB, GB, bytes, etc.)
    if config.get("normalize_sizes", False):
        # Pattern: 1024 KB, 512MB, 1.5GB, 1000 bytes, etc.
        size_pattern = r'\b\d+(?:\.\d+)?\s*(?:KB|MB|GB|TB|PB|bytes?|B)\b'
        result = re.sub(size_pattern, '<SIZE>', result, flags=re.IGNORECASE)
        # Aussi capturer les très grands nombres seuls (probablement des tailles)
        large_number_pattern = r'\b\d{6,}\b'
        result = re.sub(large_number_pattern, '<SIZE>', result)
    
    # Normalize timestamps (dates, heures, timestamps Unix, etc.)
    if config.get("normalize_timestamps", False):
        # Pattern: dates ISO (2024-01-15, 2024/01/15)
        iso_date_pattern = r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'
        result = re.sub(iso_date_pattern, '<TS>', result)
        # Pattern: heures (HH:MM:SS, HH:MM)
        time_pattern = r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?\b'
        result = re.sub(time_pattern, '<TS>', result, flags=re.IGNORECASE)
        # Pattern: timestamps Unix (10 ou 13 chiffres)
        unix_timestamp_pattern = r'\b\d{10,13}\b'
        result = re.sub(unix_timestamp_pattern, '<TS>', result)
        # Pattern: dates avec mois en lettres (Jan 15, 2024, 15-Jan-2024)
        date_with_month_pattern = r'\b\d{1,2}[-/](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-/]\d{2,4}\b'
        result = re.sub(date_with_month_pattern, '<TS>', result, flags=re.IGNORECASE)
        date_with_month_pattern2 = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b'
        result = re.sub(date_with_month_pattern2, '<TS>', result, flags=re.IGNORECASE)
    
    # Normalize file paths
    if config.get("normalize_paths", False):
        # Pattern: chemins Unix/Windows (/path/to/file, C:\path\to\file, \\server\share)
        # Format: commence par /, \, ou lettre:\, suivi de segments séparés par / ou \
        # Pattern Windows: C:\path\to\file ou \\server\share
        windows_path_pattern = r'\b(?:[A-Za-z]:)?[\\](?:[^\\/\s]+[\\])+[^\\/\s]+(?:\.[a-zA-Z0-9]+)?'
        result = re.sub(windows_path_pattern, '<PATH>', result)
        # Pattern Unix: /path/to/file
        unix_path_pattern = r'/(?:[^\\/\s]+/)+[^\\/\s]+(?:\.[a-zA-Z0-9]+)?'
        result = re.sub(unix_path_pattern, '<PATH>', result)
    
    # Normalize hashes (MD5, SHA1, SHA256, etc.)
    if config.get("normalize_hashes", False):
        # Pattern: MD5 (32 hex chars), SHA1 (40 hex chars), SHA256 (64 hex chars)
        # Format: souvent précédés de "hash:", "md5:", "sha1:", etc.
        hash_with_prefix = r'\b(?:hash|md5|sha1|sha256|sha512)\s*[:=]?\s*([0-9a-fA-F]{32,})\b'
        result = re.sub(hash_with_prefix, r'<HASH>', result, flags=re.IGNORECASE)
        # Pattern: hashes seuls (32, 40, 64 caractères hex)
        hash_standalone = r'\b[0-9a-fA-F]{32}\b|\b[0-9a-fA-F]{40}\b|\b[0-9a-fA-F]{64}\b'
        result = re.sub(hash_standalone, '<HASH>', result)
    
    # Normalize GUID/UUID
    if config.get("normalize_guids", False):
        # Pattern: GUID/UUID standard (8-4-4-4-12)
        guid_pattern = r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'
        result = re.sub(guid_pattern, '<GUID>', result)
        # Pattern: GUID sans tirets (32 hex chars)
        guid_no_dashes = r'\b[0-9a-fA-F]{32}\b'
        result = re.sub(guid_no_dashes, '<GUID>', result)
    
    # Normalize MAC addresses
    if config.get("normalize_mac_addresses", False):
        # Pattern: MAC address (XX:XX:XX:XX:XX:XX ou XX-XX-XX-XX-XX-XX)
        mac_pattern = r'\b(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}\b'
        result = re.sub(mac_pattern, '<MAC>', result)
    
    # Normalize hostnames (noms de machines)
    if config.get("normalize_hostnames", False):
        # Pattern: hostnames avec préfixe (host:, hostname:, server:, etc.)
        hostname_with_prefix = r'\b(?:host|hostname|server|machine|node)\s*[:=]?\s*[A-Za-z0-9][A-Za-z0-9._-]{2,}\b'
        result = re.sub(hostname_with_prefix, '<HOSTNAME>', result, flags=re.IGNORECASE)
        # Pattern: hostnames avec domaine (server01.example.com, host.domain.local)
        hostname_with_domain = r'\b[A-Za-z0-9][A-Za-z0-9_-]{2,}\.[A-Za-z0-9][A-Za-z0-9._-]{1,}\.[A-Za-z]{2,}\b'
        result = re.sub(hostname_with_domain, '<HOSTNAME>', result)
    
    # Normalize Associated Threat Actors
    if config.get("normalize_threat_actors", False):
        # Pattern: "Associated Threat Actor: Name" où Name peut contenir des espaces, tirets, etc.
        # Capture jusqu'à la fin de la ligne ou jusqu'à un "|" (séparateur d'informations)
        threat_actor_pattern = r'Associated Threat Actor:\s*([^|,\n]+?)(?:\s*\||\s*$)'
        result = re.sub(threat_actor_pattern, 'Associated Threat Actor: <THREAT_ACTOR>', result, flags=re.IGNORECASE)
    
    # Remove extra spaces
    if config.get("remove_extra_spaces", False):
        result = re.sub(r'\s+', ' ', result)
        result = result.strip()

    return result


def apply_label_mapping(labels: list[object], label_mapping: dict[str, str] | None) -> list[str]:
    """
    Normalise les labels et applique un mapping optionnel.
    """
    normalized = [normalize_label(l) for l in labels]
    if not label_mapping:
        return normalized
    mapping_norm = {normalize_label(k): normalize_label(v) for k, v in label_mapping.items()}
    return [mapping_norm.get(l, l) for l in normalized]


def print_label_distribution(labels: list[str], title: str) -> None:
    """Affiche la distribution des labels."""
    counts = Counter(labels)
    total = len(labels) or 1
    print(f"\n{title}")
    for lbl, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {lbl}: {cnt} ({100 * cnt / total:.2f}%)")


# ============================================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================================


def load_data_from_csv(
    train_csv_path,
    test_csv_path,
    text_field=TEXT_FIELD,
    label_field=LABEL_FIELD,
    label_mapping: dict[str, str] | None = None,
    preprocessing_config: dict | None = None,
):
    """
    Charge les données depuis les fichiers CSV.
    
    Args:
        train_csv_path: Chemin vers train.csv
        test_csv_path: Chemin vers test.csv
        text_field: Nom du champ texte
        label_field: Nom du champ label
        label_mapping: Mapping optionnel des labels
        preprocessing_config: Configuration du pré-traitement (None = pas de pré-traitement)
        
    Returns:
        sentences_train_full, labels_train_full, sentences_test, labels_test
    """
    print("Loading dataset from CSV files...")
    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)
    
    print(f"Train: {len(df_train)} samples")
    print(f"Test: {len(df_test)} samples")
    
    # Extract text and apply preprocessing
    sentences_train_full = df_train[text_field].fillna("").astype(str).tolist()
    sentences_test = df_test[text_field].fillna("").astype(str).tolist()
    
    if preprocessing_config:
        print("\n🔧 Preprocessing texts...")
        sentences_train_full = [preprocess_text(s, preprocessing_config) for s in sentences_train_full]
        sentences_test = [preprocess_text(s, preprocessing_config) for s in sentences_test]
        print(f"   Example preprocessed (train): {sentences_train_full[0][:100]}...")
        print(f"   Example preprocessed (test): {sentences_test[0][:100]}...")
    
    # Extract and map labels
    labels_train_full = apply_label_mapping(df_train[label_field].tolist(), label_mapping)
    labels_test = apply_label_mapping(df_test[label_field].tolist(), label_mapping)

    if label_mapping:
        print_label_distribution(labels_train_full, "Train label distribution (after mapping):")
        print_label_distribution(labels_test, "Test label distribution (after mapping):")
    
    return sentences_train_full, labels_train_full, sentences_test, labels_test


def split_train_dev(sentences_train_full, labels_train_full, test_size=0.2, random_state=42):
    """
    Divise le train en train et dev.
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


# ============================================================================
# MODÈLE ET TOKENIZATION
# ============================================================================


def create_tokenized_datasets(sentences_train, sentences_dev, sentences_test,
                              labels_train_encoded, labels_dev_encoded, labels_test_encoded,
                              model_name=MODEL_NAME, max_length=512):
    """
    Crée les datasets tokenisés pour HuggingFace.
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


def freeze_model_layers(model, unfreeze_last_n_layers=1):
    """
    Freeze all layers except the last N transformer layers.
    """
    # Freeze all base model parameters first
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    # Unfreeze the last N layers
    if hasattr(model.base_model, "transformer"):
        # For DistilBERT
        layers = model.base_model.transformer.layer
        num_layers = len(layers)
        
        # Unfreeze the last N layers' FFN components
        for i in range(max(0, num_layers - unfreeze_last_n_layers), num_layers):
            for name, param in layers[i].named_parameters():
                # Unfreeze FFN layers
                if "ffn.lin" in name:
                    param.requires_grad = True
    
    # Always unfreeze the classifier head
    for param in model.classifier.parameters():
        param.requires_grad = True


def create_model(model_name=MODEL_NAME, num_labels=5, unfreeze_last_n_layers=1):
    """
    Crée et configure le modèle avec freeze des couches.
    """
    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    
    # Freeze layers
    print("Freezing model layers...")
    freeze_model_layers(model, unfreeze_last_n_layers=unfreeze_last_n_layers)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model


# ============================================================================
# TRAINING
# ============================================================================


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
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


def create_stratified_sampler(labels_train_encoded):
    """
    Crée un sampler stratifié pour équilibrer les batches pendant l'entraînement.
    """
    # Compter les occurrences de chaque classe
    class_counts = Counter(labels_train_encoded)
    print(f"\nClass distribution in training set:")
    for label, count in sorted(class_counts.items()):
        print(f"  Class {label}: {count} samples ({100*count/len(labels_train_encoded):.2f}%)")
    
    # Calculer les poids pour chaque échantillon
    class_weights = {label: 1.0 / count for label, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels_train_encoded]
    
    # Créer le sampler
    from torch.utils.data import WeightedRandomSampler
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
    
    # Stratified Sampling
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
        
        def _get_train_sampler(self, train_dataset):
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
        lr_scheduler_type=LR_SCHEDULER_TYPE,
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
    """
    print("Starting training...")
    trainer.train()
    return trainer


# ============================================================================
# ÉVALUATION ET SAUVEGARDE
# ============================================================================


def evaluate_and_save_predictions(trainer, dataset_test, label_encoder, labels_test,
                                   predictions_dir="./predictions",
                                   predictions_filename="transfer_learning_predictions.csv"):
    """
    Évalue le modèle et sauvegarde les prédictions.
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
    """
    print("Saving model...")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    # Save label encoder
    label_encoder_path = f"{model_dir}/label_encoder.pkl"
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    
    print(f"✅ Modèle sauvegardé dans {model_dir}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main training function."""
    
    # Configuration du pré-traitement
    preprocessing_config = PREPROCESSING_CONFIG if ENABLE_PREPROCESSING else None
    
    # Configuration du mapping de labels
    label_mapping = LABEL_MAPPING if LABEL_MAPPING else None
    
    print("=" * 70)
    print("TRANSFER LEARNING - SIEM SEVERITY CLASSIFICATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Unfrozen layers: {UNFREEZE_LAST_N_LAYERS}")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  Label mapping: {label_mapping}")
    print(f"  Preprocessing: {ENABLE_PREPROCESSING}")
    if ENABLE_PREPROCESSING:
        print(f"  Preprocessing config: {preprocessing_config}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Batch size: {PER_DEVICE_TRAIN_BATCH_SIZE}")
    print(f"  Epochs: {NUM_TRAIN_EPOCHS}")
    print(f"  Focal loss: {USE_FOCAL_LOSS} (gamma={FOCAL_GAMMA})")
    print(f"  Weight scaling: {WEIGHT_SCALING}")
    print("=" * 70)
    
    print(f"\nDevice: {torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else 'CPU'}")
    
    # 1. Load data
    sentences_train_full, labels_train_full, sentences_test, labels_test = load_data_from_csv(
        TRAIN_CSV_PATH, 
        TEST_CSV_PATH, 
        TEXT_FIELD, 
        LABEL_FIELD, 
        label_mapping=label_mapping,
        preprocessing_config=preprocessing_config
    )
    
    # 2. Split train into train and dev
    sentences_train, sentences_dev, labels_train, labels_dev = split_train_dev(
        sentences_train_full, labels_train_full, test_size=DEV_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"Test: {len(sentences_test)}")
    
    # 3. Encode labels
    label_encoder, labels_train_encoded, labels_dev_encoded, labels_test_encoded, num_labels = encode_labels(
        labels_train, labels_dev, labels_test
    )
    
    # 4. Create tokenized datasets
    tokenizer, dataset_train, dataset_dev, dataset_test = create_tokenized_datasets(
        sentences_train, sentences_dev, sentences_test,
        labels_train_encoded, labels_dev_encoded, labels_test_encoded,
        model_name=MODEL_NAME, max_length=MAX_LENGTH
    )
    
    # 5. Create model
    model = create_model(MODEL_NAME, num_labels, unfreeze_last_n_layers=UNFREEZE_LAST_N_LAYERS)
    
    # 6. Create trainer
    trainer = create_trainer(
        model, tokenizer, dataset_train, dataset_dev,
        labels_train_encoded=labels_train_encoded,
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        logging_dir=LOGGING_DIR,
        logging_steps=LOGGING_STEPS,
        warmup_ratio=WARMUP_RATIO,
        use_focal_loss=USE_FOCAL_LOSS,
        focal_gamma=FOCAL_GAMMA,
        weight_scaling=WEIGHT_SCALING,
        use_stratified_sampling=USE_STRATIFIED_SAMPLING,
    )
    
    # 7. Train model
    trainer = train_model(trainer)
    
    # 8. Evaluate and save predictions
    predictions_df = evaluate_and_save_predictions(
        trainer,
        dataset_test,
        label_encoder,
        labels_test,
        predictions_dir=PREDICTIONS_DIR,
        predictions_filename=PREDICTIONS_FILENAME,
    )
    
    # 9. Save model
    save_model(trainer, tokenizer, label_encoder, model_dir=MODEL_DIR)
    
    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()