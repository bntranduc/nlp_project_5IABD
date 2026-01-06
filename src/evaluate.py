"""
Pipeline commune pour l'évaluation et le logging MLflow.
"""

from pathlib import Path
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)


def compute_metrics(y_true, y_pred):
    """
    Calcule les métriques standardisées pour tous les modèles.
    
    Args:
        y_true: Labels réels
        y_pred: Labels prédits
        
    Returns:
        Dictionary avec toutes les métriques
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    
    return metrics


def run_evaluate(predictions_csv_path, experiment_name, run_name=None):
    """
    Charge les prédictions depuis CSV, calcule les métriques et les log dans MLflow.
    Pipeline commune utilisée pour évaluer tous les modèles.
    
    Args:
        predictions_csv_path: Chemin vers le fichier CSV contenant les prédictions
                              Doit contenir les colonnes: 'y_true' et 'y_pred'
        experiment_name: Nom de l'expérience MLflow
        run_name: Nom optionnel pour le run (default: nom du fichier)
        
    Returns:
        Dictionary avec toutes les métriques calculées
    """
    # Charger les prédictions
    print(f"Chargement des prédictions depuis {predictions_csv_path}...")
    df = pd.read_csv(predictions_csv_path)
    
    # Vérifier que les colonnes nécessaires existent
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError(
            f"Le fichier CSV doit contenir les colonnes 'y_true' et 'y_pred'. "
            f"Colonnes trouvées: {df.columns.tolist()}"
        )
    
    y_true = df["y_true"]
    y_pred = df["y_pred"]
    
    print(f"Échantillons chargés: {len(y_true)}")
    
    # Calculer les métriques
    print("Calcul des métriques...")
    metrics = compute_metrics(y_true, y_pred)
    
    # Afficher les métriques
    print("\n📊 Métriques calculées:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Logger dans MLflow
    print(f"\n📝 Logging dans MLflow (experiment: {experiment_name})...")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        # Logger les métriques
        mlflow.log_metrics(metrics)
        
        # Logger le chemin du fichier de prédictions comme artefact
        mlflow.log_artifact(predictions_csv_path, "predictions")
        
        # Logger les paramètres
        mlflow.log_params({
            "predictions_file": str(predictions_csv_path),
            "n_samples": len(y_true),
        })
        
        # Créer et logger un rapport de classification
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = Path(predictions_csv_path).parent / "classification_report.csv"
        report_df.to_csv(report_path)
        mlflow.log_artifact(str(report_path), "reports")
        
        # Créer et logger la matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm)
        cm_path = Path(predictions_csv_path).parent / "confusion_matrix.csv"
        cm_df.to_csv(cm_path)
        mlflow.log_artifact(str(cm_path), "reports")
        
        print(f"✅ Métriques loggées dans MLflow (run: {mlflow.active_run().info.run_id})")
    
    return metrics


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python3 src/evaluate.py <predictions_csv_path> <experiment_name> [run_name]")
        print("\nExemple:")
        print("  python3 src/evaluate.py ./predictions/random_forest_predictions.csv siem_severity_classification random_forest")
        sys.exit(1)
    
    predictions_path = sys.argv[1]
    experiment_name = sys.argv[2]
    run_name = sys.argv[3] if len(sys.argv) > 3 else None
    
    run_evaluate(predictions_path, experiment_name, run_name)
