"""
Few-shot learner utilisant l'API OpenAI pour la classification de sévérité SIEM.
"""

import os
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from openai import OpenAI
from src.data import load_and_prepare

from dotenv import load_dotenv



# Configuration
TEXT_FIELD = "description"
LABEL_FIELD = "severity"
MODEL_NAME = "openai/gpt-oss-20b" 
NUM_FEW_SHOT_EXAMPLES = 5  # Nombre d'exemples à utiliser pour le few-shot
BATCH_SIZE = 10  # Nombre de logs à classifier par requête API
VALID_LABELS = ["info", "low", "medium", "high", "critical", "emergency"]  # Labels valides
DEBUG = True  # Activer pour voir les réponses brutes de l'API


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Crée un client OpenAI.
    
    Args:
        api_key: Clé API OpenAI (si None, utilise OPENAI_API_KEY env var)
        
    Returns:
        Client OpenAI
    """

    load_dotenv()
    api_key = os.getenv('OPENAI_KEY')
    

    
    return OpenAI(     
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )


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


def create_prompt(few_shot_examples: List[Tuple[str, str]], texts: List[str]) -> str:
    """
    Crée le prompt pour l'API OpenAI avec les exemples few-shot.
    
    Args:
        few_shot_examples: Liste d'exemples (texte, label)
        texts: Liste de textes à classifier
        
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
    
    # Ajouter les textes à classifier
    if len(texts) == 1:
        prompt += f"Maintenant, classifie cet événement:\n"
        prompt += f"Description: {texts[0]}\n"
        prompt += f"Sévérité: "
    else:
        prompt += f"Maintenant, classifie ces {len(texts)} événements.\n\n"
        for i, text in enumerate(texts, 1):
            prompt += f"Événement {i}:\n"
            prompt += f"Description: {text}\n\n"
        prompt += f"IMPORTANT: Réponds UNIQUEMENT avec les {len(texts)} labels de sévérité, un par ligne, dans le même ordre que les événements. Chaque ligne doit contenir uniquement le label (info, low, medium, high, critical, ou emergency), sans numéro, sans texte supplémentaire.\n"
        prompt += f"Format attendu:\n"
        for i in range(len(texts)):
            prompt += f"<label>\n"
    
    return prompt


def clean_and_validate_label(label: str, valid_labels: List[str] = VALID_LABELS) -> str:
    """
    Nettoie et valide un label.
    
    Args:
        label: Label brut à nettoyer
        valid_labels: Liste des labels valides
        
    Returns:
        Label nettoyé et validé, ou "unknown" si invalide
    """
    if not label:
        return "unknown"
    
    # Nettoyer le label
    label_original = label
    label = label.strip().lower()
    
    # Enlever la ponctuation en début/fin
    label = label.strip('.,!?;:()[]{}"\'').strip()
    
    # Enlever les préfixes numériques (ex: "1. high", "1-high", "1) high")
    if label:
        # Pattern: "1. label" ou "1) label" ou "1-label"
        label = re.sub(r'^\d+[\.\)\-\s]+', '', label).strip()
    
    # Enlever les préfixes de texte (ex: "Événement 1: high", "Event 1: high", "Sévérité: high")
    if ':' in label:
        # Prendre la partie après le dernier ":"
        parts = label.split(':')
        label = parts[-1].strip()
    
    # Enlever les mots-clés communs qui peuvent précéder le label
    prefixes_to_remove = ['severity', 'sévérité', 'label', 'classification', 'class', 'event', 'événement']
    for prefix in prefixes_to_remove:
        if label.startswith(prefix):
            label = label[len(prefix):].strip(' :-\t').strip()
    
    # Vérifier si le label est valide (exact match)
    if label in valid_labels:
        return label
    
    # Chercher un label valide dans le texte (mot entier)
    # Utiliser des word boundaries pour éviter les faux positifs
    for valid_label in valid_labels:
        # Chercher le label comme mot entier
        pattern = r'\b' + re.escape(valid_label) + r'\b'
        if re.search(pattern, label):
            return valid_label
    
    # Si rien ne correspond, essayer de trouver n'importe quel label valide dans le texte original
    label_lower = label_original.lower()
    for valid_label in valid_labels:
        if valid_label in label_lower:
            return valid_label
    
    # Si rien ne correspond, retourner unknown
    if DEBUG:
        print(f"[DEBUG] Label non reconnu: '{label_original}' -> '{label}'")
    return "unknown"


def predict_with_openai(client: OpenAI, prompt: str, num_predictions: int = 1, 
                        model: str = MODEL_NAME, max_retries: int = 3) -> List[str]:
    """
    Fait des prédictions avec l'API OpenAI.
    
    Args:
        client: Client OpenAI
        prompt: Prompt à envoyer
        num_predictions: Nombre de prédictions attendues
        model: Nom du modèle à utiliser
        max_retries: Nombre maximum de tentatives en cas d'erreur
        
    Returns:
        Liste des labels prédits
    """
    for attempt in range(max_retries):
        try:
            # Ajuster max_tokens en fonction du nombre de prédictions
            max_tokens = max(10, num_predictions * 20)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un expert en cybersécurité qui classifie la sévérité d'événements SIEM. Réponds UNIQUEMENT avec le(s) label(s) de sévérité (info, low, medium, high, critical, emergency), sans explication, sans numéro, sans texte supplémentaire. Si plusieurs événements, un label par ligne."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Faible température pour plus de cohérence
                max_tokens=max_tokens,
            )
            
            response_text = response.choices[0].message.content.strip()
            
            if DEBUG:
                print(f"\n[DEBUG] Réponse brute de l'API ({num_predictions} prédictions attendues):")
                print(f"'{response_text}'")
                print(f"Lignes: {response_text.split(chr(10))}")
            
            if num_predictions == 1:
                # Un seul résultat
                prediction = clean_and_validate_label(response_text)
                return [prediction]
            else:
                # Plusieurs résultats - parser les lignes
                lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                
                if DEBUG:
                    print(f"[DEBUG] Lignes parsées: {lines}")
                
                predictions = []
                
                # Prendre les premières lignes qui correspondent au nombre attendu
                for line in lines:
                    if len(predictions) >= num_predictions:
                        break
                    cleaned = clean_and_validate_label(line)
                    predictions.append(cleaned)
                
                # Si on n'a pas assez de prédictions, compléter avec "unknown"
                while len(predictions) < num_predictions:
                    predictions.append("unknown")
                
                if DEBUG:
                    print(f"[DEBUG] Prédictions finales: {predictions}")
                
                return predictions[:num_predictions]
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Erreur (tentative {attempt + 1}/{max_retries}): {e}")
                print(f"Attente {wait_time} secondes avant de réessayer...")
                time.sleep(wait_time)
            else:
                print(f"Erreur après {max_retries} tentatives: {e}")
                if DEBUG:
                    print(f"[DEBUG] Retour de valeurs par défaut: unknown")
                return ["unknown"] * num_predictions  # Valeur par défaut en cas d'échec
    
    return ["unknown"] * num_predictions


def predict_batch(client: OpenAI, texts: List[str], few_shot_examples: List[Tuple[str, str]], 
                  model: str = MODEL_NAME, batch_size: int = BATCH_SIZE, delay: float = 0.1) -> List[str]:
    """
    Fait des prédictions sur un batch de textes en traitant plusieurs logs par requête.
    
    Args:
        client: Client OpenAI
        texts: Liste de textes à classifier
        few_shot_examples: Exemples few-shot
        model: Nom du modèle
        batch_size: Nombre de logs à traiter par requête API
        delay: Délai entre les requêtes (pour éviter les rate limits)
        
    Returns:
        Liste des labels prédits
    """
    predictions = []
    total = len(texts)
    num_batches = (total + batch_size - 1) // batch_size  # Arrondi supérieur
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total)
        batch_texts = texts[start_idx:end_idx]
        
        # Créer le prompt pour ce batch
        prompt = create_prompt(few_shot_examples, batch_texts)
        
        # Faire la prédiction pour tout le batch
        batch_predictions = predict_with_openai(
            client, 
            prompt, 
            num_predictions=len(batch_texts),
            model=model
        )
        
        predictions.extend(batch_predictions)
        
        # Afficher le progrès
        processed = end_idx
        print(f"Progrès: {processed}/{total} ({100*processed/total:.1f}%) - Batch {batch_idx + 1}/{num_batches}")
        
        # Délai pour éviter les rate limits (sauf pour le dernier batch)
        if batch_idx < num_batches - 1:
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

    X_test = X_test.head(100)
    y_test = y_test.head(100)
    
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
    print(f"Traitement par batch de {BATCH_SIZE} logs...")
    if DEBUG:
        print("⚠️  Mode DEBUG activé - les réponses brutes seront affichées")
    print("⚠️  Cela peut prendre du temps...")
    predictions = predict_batch(
        client,
        X_test.tolist(),
        few_shot_examples,
        model=MODEL_NAME,
        batch_size=BATCH_SIZE,
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

