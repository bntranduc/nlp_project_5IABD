#!/bin/bash
# Script pour lancer l'interface UI de MLflow

echo "🚀 Démarrage de MLflow UI..."
echo ""
echo "L'interface sera accessible sur: http://localhost:5000"
echo "Appuyez sur Ctrl+C pour arrêter le serveur"
echo ""

# Lancer MLflow UI
# --backend-store-uri spécifie où sont stockées les données (par défaut: ./mlruns)
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000


