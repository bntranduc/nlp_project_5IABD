"""
Script simple pour entraîner un Random Forest sur le dataset SIEM.
Utilise uniquement description comme feature et prédit la sévérité.
Sauvegarde les prédictions dans un fichier CSV pour évaluation commune.
"""

import os
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data import load_and_prepare


def main():
    """Fonction principale."""
    
    # Chemins des fichiers CSV
    train_csv_path = "./data/raw/SIEM_Dataset/train.csv"
    test_csv_path = "./data/raw/SIEM_Dataset/test.csv"
    
    # Charger et préparer les données avec la pipeline commune
    X_train, y_train, X_test, y_test = load_and_prepare(
        train_csv_path,
        test_csv_path,
        text_field="description",
        label_field="severity"
    )
    
    # Vectoriser le texte (TF-IDF) 
    print("\nVectorisation du texte (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Features: {X_train_vec.shape[1]}")
    
    # Entraîner Random Forest
    print("\nEntraînement du Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_vec, y_train)
    
    # Prédire
    print("Prédiction sur le test...")
    y_pred = rf.predict(X_test_vec)
    
    # Sauvegarder les prédictions dans un CSV
    predictions_dir = "./predictions"
    os.makedirs(predictions_dir, exist_ok=True)
    
    predictions_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred
    })
    
    predictions_path = f"{predictions_dir}/random_forest_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\n✅ Prédictions sauvegardées dans {predictions_path}")
    print(f"   Échantillons: {len(predictions_df)}")


if __name__ == "__main__":
    main()

