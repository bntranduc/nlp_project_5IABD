"""
Few-shot learner utilisant l'API OpenAI pour la classification de sévérité SIEM.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from openai import OpenAI
from src.data import load_and_prepare


# Configuration
TEXT_FIELD = "description"
LABEL_FIELD = "severity"
MODEL_NAME = "gpt-3.5-turbo"  # Modèle léger
NUM_FEW_SHOT_EXAMPLES = 5  # Nombre d'exemples à utiliser pour le few-shot


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Crée un client OpenAI.
    
    Args:
        api_key: Clé API OpenAI (si None, utilise OPENAI_API_KEY env var)
        
    Returns:
        Client OpenAI
    """
    api_key = "YOUR_KEY_HERE"


    
    return OpenAI(api_key=api_key)


def select_few_shot_examples(X_train, y_train, num_examples: int = NUM_FEW_SHOT_EXAMPLES) -> List[Tuple[str, str]]:
    """
    Sélectionne des exemples représentatifs pour le few-shot learning.
    Essaie d'avoir au moins un exemple de chaque classe.
    
    Args:
        X_train: Textes d'entraînement
        y_train: Labels d'entraînement
        num_examples: Nombre d'exemples à sélectionner
        
    Returns:
        Liste de tuples (texte, label)
    """
    df = pd.DataFrame({"text": X_train, "label": y_train})
    
    # Obtenir toutes les classes uniques
    unique_labels = df["label"].unique()
    
    examples = []
    
    # Prendre au moins un exemple de chaque classe
    for label in unique_labels:
        label_examples = df[df["label"] == label].head(1)
        if len(label_examples) > 0:
            examples.append((label_examples.iloc[0]["text"], label_examples.iloc[0]["label"]))
    
    # Compléter avec des exemples aléatoires jusqu'à num_examples
    remaining = num_examples - len(examples)
    if remaining > 0:
        remaining_examples = df.sample(n=min(remaining, len(df)), random_state=42)
        for _, row in remaining_examples.iterrows():
            if len(examples) >= num_examples:
                break
            examples.append((row["text"], row["label"]))
    
    return examples[:num_examples]


def create_prompt(few_shot_examples: List[Tuple[str, str]], text: str) -> str:
    """
    Crée le prompt pour l'API OpenAI avec les exemples few-shot.
    
    Args:
        few_shot_examples: Liste d'exemples (texte, label)
        text: Texte à classifier
        
    Returns:
        Prompt formaté
    """
    # Obtenir toutes les classes possibles
    all_labels = list(set([label for _, label in few_shot_examples]))
    all_labels.sort()
    
    prompt = f"""Tu es un expert en cybersécurité. Ta tâche est de classifier la sévérité d'événements de sécurité SIEM.

Les niveaux de sévérité possibles sont: {', '.join(all_labels)}.

Voici quelques exemples:

"""
    
    # Ajouter les exemples few-shot
    for i, (example_text, example_label) in enumerate(few_shot_examples, 1):
        prompt += f"Exemple {i}:\n"
        prompt += f"Description: {example_text}\n"
        prompt += f"Sévérité: {example_label}\n\n"
    
    # Ajouter le texte à classifier
    prompt += f"Maintenant, classifie cet événement:\n"
    prompt += f"Description: {text}\n"
    prompt += f"Sévérité: "
    
    return prompt


def predict_with_openai(client: OpenAI, prompt: str, model: str = MODEL_NAME, max_retries: int = 3) -> str:
    """
    Fait une prédiction avec l'API OpenAI.
    
    Args:
        client: Client OpenAI
        prompt: Prompt à envoyer
        model: Nom du modèle à utiliser
        max_retries: Nombre maximum de tentatives en cas d'erreur
        
    Returns:
        Label prédit
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un expert en cybersécurité qui classifie la sévérité d'événements SIEM. Réponds uniquement avec le label de sévérité, sans explication."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Faible température pour plus de cohérence
                max_tokens=10,  # Les labels sont courts
            )
            
            prediction = response.choices[0].message.content.strip()
            
            # Nettoyer la prédiction (enlever les points, espaces, etc.)
            prediction = prediction.strip('.,!?;:').strip()
            
            return prediction
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Erreur (tentative {attempt + 1}/{max_retries}): {e}")
                print(f"Attente {wait_time} secondes avant de réessayer...")
                time.sleep(wait_time)
            else:
                print(f"Erreur après {max_retries} tentatives: {e}")
                return "unknown"  # Valeur par défaut en cas d'échec
    
    return "unknown"


def predict_batch(client: OpenAI, texts: List[str], few_shot_examples: List[Tuple[str, str]], 
                  model: str = MODEL_NAME, delay: float = 0.1) -> List[str]:
    """
    Fait des prédictions sur un batch de textes.
    
    Args:
        client: Client OpenAI
        texts: Liste de textes à classifier
        few_shot_examples: Exemples few-shot
        model: Nom du modèle
        delay: Délai entre les requêtes (pour éviter les rate limits)
        
    Returns:
        Liste des labels prédits
    """
    predictions = []
    total = len(texts)
    
    for i, text in enumerate(texts, 1):
        prompt = create_prompt(few_shot_examples, text)
        prediction = predict_with_openai(client, prompt, model)
        predictions.append(prediction)
        
        if i % 10 == 0:
            print(f"Progrès: {i}/{total} ({100*i/total:.1f}%)")
        
        # Délai pour éviter les rate limits
        if i < total:
            time.sleep(delay)
    
    return predictions


def main():
    """Fonction principale."""
    
    # Charger les données
    print("Chargement des données...")
    X_train, y_train, X_test, y_test = load_and_prepare(
        "./data/raw/SIEM_Dataset/train.csv",
        "./data/raw/SIEM_Dataset/test.csv",
        text_field=TEXT_FIELD,
        label_field=LABEL_FIELD
    )
    
    print(f"Train: {len(X_train)} échantillons")
    print(f"Test: {len(X_test)} échantillons")
    
    # Sélectionner les exemples few-shot
    few_shot_examples = select_few_shot_examples(X_train, y_train, NUM_FEW_SHOT_EXAMPLES)
    print(f"Exemples few-shot: {len(few_shot_examples)}")
    
    # Créer le client OpenAI
    client = get_openai_client()
    print(f"Modèle: {MODEL_NAME}")
    
    # Faire les prédictions sur TOUS les échantillons
    print(f"\nPrédiction sur {len(X_test)} échantillons...")
    print("⚠️  Cela peut prendre du temps...")
    predictions = predict_batch(
        client,
        X_test.tolist(),
        few_shot_examples,
        model=MODEL_NAME,
        delay=0.1
    )
    
    # Sauvegarder
    os.makedirs("./predictions", exist_ok=True)
    predictions_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": predictions
    })
    predictions_df.to_csv("./predictions/few_shot_learner_predictions.csv", index=False)
    
    # Résultat
    accuracy = (predictions_df["y_true"] == predictions_df["y_pred"]).mean()
    print(f"\n✅ Accuracy: {accuracy:.2%}")
    print(f"✅ Sauvegardé dans ./predictions/few_shot_learner_predictions.csv")
    print(f"   Total: {len(predictions_df)} échantillons")


if __name__ == "__main__":
    main()

