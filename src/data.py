"""
Pipeline commune pour le chargement et la préparation des données.
"""

import pandas as pd


def load_and_prepare(train_csv_path, test_csv_path, text_field="description", label_field="severity"):
    """
    Charge les fichiers train.csv et test.csv et prépare les données.
    Pipeline commune utilisée par tous les modèles.
    
    Args:
        train_csv_path: Chemin vers le fichier train.csv
        test_csv_path: Chemin vers le fichier test.csv
        text_field: Nom du champ contenant le texte (default: "description")
        label_field: Nom du champ contenant les labels (default: "severity")
        
    Returns:
        X_train, y_train, X_test, y_test (pandas Series)
    """
    # Charger les fichiers CSV
    print(f"Chargement de {train_csv_path}...")
    df_train = pd.read_csv(train_csv_path)
    print(f"Train: {len(df_train)} échantillons")
    
    print(f"Chargement de {test_csv_path}...")
    df_test = pd.read_csv(test_csv_path)
    print(f"Test: {len(df_test)} échantillons")
    
    # Extraire les champs nécessaires
    X_train = df_train[text_field].fillna("").astype(str)
    y_train = df_train[label_field]
    
    X_test = df_test[text_field].fillna("").astype(str)
    y_test = df_test[label_field]
    
    print(f"\nDonnées préparées:")
    print(f"  Train: {len(X_train)} échantillons")
    print(f"  Test: {len(X_test)} échantillons")
    
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Test
    X_train, y_train, X_test, y_test = load_and_prepare(
        "./data/raw/SIEM_Dataset/train.csv",
        "./data/raw/SIEM_Dataset/test.csv"
    )
    print(X_train.head())
    print(y_train.head())
