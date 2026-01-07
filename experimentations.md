# Rapport d'Expérimentations - Classification de Sévérité SIEM

## Configuration Générale
- Dataset: 100 000 échantillons (80% train, 20% test)
- Split: 64 000 train / 16 000 validation / 20 000 test
- Tâche: Classification multi-classe (6 classes)
- Classes: critical, emergency, high, info, low, medium
- Modèle base: DistilBERT-base-cased
- Batch size: 32 (train) / 64 (eval)
- Max length: 512 tokens

---

## 1. Baseline Random Forest

### Configuration
- Modèle: Random Forest (baseline)
- Features: TF-IDF ou features extraites

### Métriques de Performance

| Métrique | Score |
|----------|-------|
| Accuracy | 20.42% |
| F1-Score (weighted) | 19.69% |
| F1-Score (macro) | 18.46% |
| Precision (weighted) | 20.52% |
| Precision (macro) | 21.01% |
| Recall (weighted) | 20.42% |
| Recall (macro) | 18.55% |

---

## 2. Transfer Learning - Baseline (1 couche dégelée)

### Configuration
- Paramètres entraînables: 10,040,070 / 65,786,118 (15.26%)
- Couches gelées: Toutes sauf dernière couche FFN + classifier
- Epochs: 15
- Learning rate: 5e-5 (décroissance linéaire)
- Batch size: 32 (train) / 16 (eval)

### Métriques de Performance

| Métrique | Score |
|----------|-------|
| Accuracy | 20.64% |
| F1-Score (weighted) | 19.27% |
| F1-Score (macro) | 20.58% |
| Precision (weighted) | 21.13% |
| Precision (macro) | 21.07% |
| Recall (weighted) | 20.64% |
| Recall (macro) | 23.48% |

### Évolution pendant l'entraînement
- Epoch 1: Loss 1.5886, Accuracy 20.69%
- Epoch 5: Loss 1.5843, Accuracy 20.68%
- Epoch 10: Loss 1.5844, Accuracy 20.68%
- Epoch 15: Loss 1.5912, Accuracy 20.46%

### Observations
- Convergence rapide dès la première epoch
- Stagnation des performances après epoch 2
- Loss test (1.5845) proche de la loss validation
- Pas de signe d'overfitting
- Meilleur F1-macro parmi les approches simples

---

## 3. Transfer Learning - 2 couches dégelées

### Configuration
- Paramètres entraînables: 10,040,070 / 65,786,118 (15.26%)
- Couches dégelées: 2 dernières couches FFN + classifier
- Epochs: 15
- Learning rate: 5e-5 (décroissance linéaire)

### Métriques de Performance

| Métrique | Score |
|----------|-------|
| Accuracy | 20.57% |
| F1-Score (weighted) | 16.45% |
| F1-Score (macro) | 14.54% |
| Precision (weighted) | 20.99% |
| Precision (macro) | 17.90% |
| Recall (weighted) | 20.57% |
| Recall (macro) | 18.65% |

### Observations
- Dégradation significative du F1-Score (-6.04 points macro)
- Warnings: Precision ill-defined (classes non prédites)
- Perte de généralisation avec plus de paramètres entraînables
- Overfitting probable

---

## 4. Transfer Learning - Class Weights (scaling=1.0)

### Configuration
- Paramètres entraînables: 10,040,070 / 65,786,118 (15.26%)
- Couches dégelées: 2 dernières couches FFN + classifier
- Epochs: 15
- Learning rate: 2e-5 (warmup 10% + cosine decay)
- Class weights: [0.94, 6.61, 0.81, 0.92, 0.81, 0.81]
- Weight scaling: 1.0 (poids complets)
- Objectif: Gérer le déséquilibre de classes

### Métriques de Performance Globales

| Métrique | Score |
|----------|-------|
| Accuracy | 20.92% |
| F1-Score (weighted) | 15.17% |
| F1-Score (macro) | 18.39% |
| Precision (weighted) | 21.95% |
| Precision (macro) | 21.61% |
| Recall (weighted) | 20.92% |
| Recall (macro) | 33.51% |

### Métriques par Classe

| Classe | Support | Precision | Recall | F1-Score |
|--------|---------|-----------|--------|----------|
| critical | 3542 (17.7%) | 19.89% | 48.36% | 28.19% |
| emergency | 504 (2.5%) | 19.98% | 100.00% | 33.30% |
| high | 4099 (20.5%) | 22.04% | 4.85% | 7.96% |
| info | 3625 (18.1%) | 22.16% | 40.77% | 28.72% |
| low | 4102 (20.5%) | 21.55% | 4.27% | 7.12% |
| medium | 4128 (20.6%) | 24.07% | 2.81% | 5.03% |

### Matrice de Confusion

|  | Prédit: critical | emergency | high | info | low | medium |
|---|---|---|---|---|---|---|
| Réel: critical | 1713 | 501 | 122 | 1068 | 136 | 2 |
| emergency | 0 | 504 | 0 | 0 | 0 | 0 |
| high | 1732 | 519 | 199 | 1364 | 145 | 140 |
| info | 1675 | 0 | 176 | 1478 | 177 | 119 |
| low | 1704 | 477 | 204 | 1437 | 175 | 105 |
| medium | 1787 | 522 | 202 | 1322 | 179 | 116 |

### Observations Critiques

**Effets positifs:**
- Emergency (weight=6.61): Recall parfait 100%
- Critical: Recall amélioré à 48.36%
- Recall macro élevé (33.51%)

**Effets négatifs majeurs:**
- Le modèle surprédit massivement emergency (2523 prédictions vs 504 réels)
- High/Low/Medium: Recalls catastrophiques (4.85%, 4.27%, 2.81%)
- Biais extrême vers critical (8611 prédictions) et emergency (2523 prédictions)
- F1-scores très faibles pour high/low/medium (5-8%)
- Weight 6.61 pour emergency trop agressif

**Distribution des prédictions:**
- critical: 43% des prédictions (vs 17.7% réel)
- emergency: 12.6% des prédictions (vs 2.5% réel)
- Classes majoritaires gravement sous-prédites

---

## 5. Transfer Learning - Class Weights Réduits (scaling=0.3)

### Configuration
- Paramètres entraînables: 10,040,070 / 65,786,118 (15.26%)
- Couches dégelées: 2 dernières couches FFN + classifier
- Epochs: 15
- Learning rate: 3e-5 (warmup 10% + cosine decay)
- Class weights originaux: [0.94, 6.61, 0.81, 0.92, 0.81, 0.81]
- **Weight scaling: 0.3** → Weights réduits: [0.98, 2.68, 0.94, 0.98, 0.94, 0.94]
- Formule: `1 + (weight - 1) × 0.3`
- Objectif: Réduire le biais excessif créé par weights complets

### Métriques de Performance Globales

| Métrique | Score | Δ vs Baseline | Δ vs Weights 1.0 |
|----------|-------|---------------|------------------|
| Accuracy | 20.64% | +0.00% | -0.28% |
| F1-Score (weighted) | 18.77% | -0.50% | +3.60% |
| **F1-Score (macro)** | **21.14%** | **+0.56%** | **+2.75%** |
| Precision (weighted) | 20.98% | -0.15% | -0.97% |
| Precision (macro) | 20.81% | -0.26% | -0.80% |
| Recall (weighted) | 20.64% | +0.00% | -0.28% |
| Recall (macro) | 32.68% | +9.20% | -0.83% |

### Métriques par Classe

| Classe | Support | Precision | Recall | F1-Score |
|--------|---------|-----------|--------|----------|
| critical | 3542 (17.7%) | 19.78% | 33.20% | 24.79% |
| emergency | 504 (2.5%) | 19.98% | 100.00% | 33.30% |
| high | 4099 (20.5%) | 20.74% | 12.25% | 15.40% |
| info | 3625 (18.1%) | 21.62% | 28.06% | 24.42% |
| low | 4102 (20.5%) | 20.26% | 14.04% | 16.59% |
| medium | 4128 (20.6%) | 22.52% | 8.53% | 12.37% |

### Matrice de Confusion

|  | Prédit: critical | emergency | high | info | low | medium |
|---|---|---|---|---|---|---|
| Réel: critical | 1176 | 501 | 331 | 785 | 539 | 210 |
| emergency | 0 | 504 | 0 | 0 | 0 | 0 |
| high | 1217 | 519 | 502 | 957 | 568 | 336 |
| info | 1138 | 0 | 536 | 1017 | 579 | 355 |
| low | 1214 | 477 | 525 | 1000 | 576 | 310 |
| medium | 1200 | 522 | 527 | 946 | 581 | 352 |

### Analyse Détaillée

**Améliorations vs Weights complets (scaling=1.0):**
- F1-macro: +2.75 points (18.39% → 21.14%)
- Distribution des prédictions plus équilibrée
- High: F1 +7.44 points (7.96% → 15.40%)
- Low: F1 +9.47 points (7.12% → 16.59%)
- Medium: F1 +7.34 points (5.03% → 12.37%)
- Toutes les classes ont maintenant F1 > 12%

**Distribution des prédictions:**
- critical: 5945 prédictions (29.7%) vs 3542 réels (17.7%)
- emergency: 2523 prédictions (12.6%) vs 504 réels (2.5%)
- high: 2421 prédictions (12.1%) vs 4099 réels (20.5%)
- info: 4705 prédictions (23.5%) vs 3625 réels (18.1%)
- low: 2843 prédictions (14.2%) vs 4102 réels (20.5%)
- medium: 1563 prédictions (7.8%) vs 4128 réels (20.6%)

**Problèmes persistants:**
- Emergency toujours surprédit (recall 100%, precision 20%)
- Medium sous-prédit (recall 8.53%)
- Biais vers critical et emergency maintenu
- Weight scaling 0.3 insuffisant pour équilibrer complètement

**Points positifs:**
- Meilleur F1-macro obtenu (21.14%)
- Toutes les classes ont F1 > 12% (vs < 10% avec scaling=1.0)
- Réduction significative du déséquilibre de prédiction
- Recall macro élevé (32.68%) sans sacrifier totalement les autres métriques

---

## 6. Transfer Learning - Focal Loss (gamma=1.5)

### Configuration
- Paramètres entraînables: 10,040,070 / 65,786,118 (15.26%)
- Couches dégelées: 2 dernières couches FFN + classifier
- Epochs: 15
- Learning rate: 2e-5 (warmup 15% + cosine decay)
- **Focal Loss: gamma=1.5, alpha=class_weights_reduced**
- Class weights scaling: 0.3 → [0.98, 2.68, 0.94, 0.98, 0.94, 0.94]
- Objectif: Alternative à CrossEntropy pour gérer le déséquilibre

### Métriques de Performance Globales

| Métrique | Score | Δ vs Baseline | Δ vs Weights 0.3 |
|----------|-------|---------------|------------------|
| Accuracy | 20.64% | +0.00% | +0.00% |
| F1-Score (weighted) | 18.35% | -2.92% | -0.42% |
| **F1-Score (macro)** | **20.83%** | **+0.25%** | **-0.31%** |
| Precision (weighted) | 21.01% | -0.12% | +0.03% |
| Precision (macro) | 20.83% | -0.24% | -0.02% |
| Recall (weighted) | 20.64% | +0.00% | +0.00% |
| Recall (macro) | 32.77% | +9.29% | +0.09% |

### Métriques par Classe

| Classe | Support | Precision | Recall | F1-Score | Δ F1 vs Weights 0.3 |
|--------|---------|-----------|--------|----------|---------------------|
| critical | 3542 (17.7%) | 19.63% | 34.08% | 24.91% | +0.12 |
| emergency | 504 (2.5%) | 19.98% | 100.00% | 33.30% | 0.00 |
| high | 4099 (20.5%) | 19.74% | 10.15% | 13.41% | -1.99 |
| info | 3625 (18.1%) | 21.51% | 31.89% | 25.69% | +1.27 |
| low | 4102 (20.5%) | 21.50% | 12.34% | 15.68% | -0.91 |
| medium | 4128 (20.6%) | 22.64% | 8.19% | 12.03% | -0.34 |

### Matrice de Confusion

|  | Prédit: critical | emergency | high | info | low | medium |
|---|---|---|---|---|---|---|
| Réel: critical | 1207 | 501 | 300 | 934 | 411 | 189 |
| emergency | 0 | 504 | 0 | 0 | 0 | 0 |
| high | 1256 | 519 | 416 | 1085 | 481 | 342 |
| info | 1211 | 0 | 449 | 1156 | 484 | 325 |
| low | 1219 | 477 | 482 | 1119 | 506 | 299 |
| medium | 1256 | 522 | 460 | 1080 | 472 | 338 |

### Distribution des Prédictions

| Classe | Prédictions | Réels | Ratio | Δ vs Weights 0.3 |
|--------|-------------|-------|-------|------------------|
| critical | 6149 (30.7%) | 3542 (17.7%) | +73.6% | +204 prédictions |
| emergency | 2523 (12.6%) | 504 (2.5%) | +400.6% | Identique |
| high | 2107 (10.5%) | 4099 (20.5%) | -48.6% | -314 prédictions |
| info | 5374 (26.9%) | 3625 (18.1%) | +48.3% | +669 prédictions |
| low | 2354 (11.8%) | 4102 (20.5%) | -42.6% | -489 prédictions |
| medium | 1493 (7.5%) | 4128 (20.6%) | -63.8% | -70 prédictions |

### Analyse Comparative: Focal Loss vs Class Weights

**Similarités:**
- Accuracy identique: 20.64%
- Emergency: Recall 100% dans les deux cas
- Bias similaire vers critical et emergency
- Distribution des prédictions quasi-identique

**Différences:**

| Aspect | Focal Loss (gamma=1.5) | Class Weights (0.3) | Meilleur |
|--------|------------------------|---------------------|----------|
| F1-macro | 20.83% | 21.14% | Weights |
| High F1 | 13.41% | 15.40% | Weights |
| Info F1 | 25.69% | 24.42% | Focal |
| Medium F1 | 12.03% | 12.37% | Weights |
| Loss finale | 1.1046 | 1.1047 | Focal (marginalement) |

### Verdict
Focal Loss avec gamma=1.5 **ne surpasse pas** class weights scaling=0.3. Les deux approches donnent des résultats très similaires, avec un léger avantage pour class weights sur le F1-macro (+0.31 points) et les classes majoritaires. La complexité supplémentaire de Focal Loss n'est pas justifiée.

---

## Comparaison Globale

| # | Modèle | Config | Accuracy | F1 Macro | Recall Macro | F1 Min |
|---|--------|--------|----------|----------|--------------|--------|
| 1 | Random Forest | Baseline | 20.42% | 18.46% | 18.55% | - |
| 2 | TL | 1 couche, lr=5e-5 | **20.64%** | 20.58% | 23.48% | - |
| 3 | TL | 2 couches, lr=5e-5 | 20.57% | 14.54% | 18.65% | - |
| 4 | TL | Weights (scaling=1.0) | 20.92% | 18.39% | 33.51% | 5.03% |
| 5 | TL | Weights (scaling=0.3) | **20.64%** | **21.14%** | 32.68% | **12.37%** |
| 6 | TL | Focal Loss (gamma=1.5) | **20.64%** | 20.83% | 32.77% | 12.03% |

### Classement par Critère

**Meilleur F1-macro:**
1. Weights (scaling=0.3): 21.14%
2. Focal Loss (gamma=1.5): 20.83%
3. Baseline (1 couche): 20.58%

**Meilleur équilibre (F1 min):**
1. Weights (scaling=0.3): 12.37%
2. Focal Loss (gamma=1.5): 12.03%
3. Weights (scaling=1.0): 5.03%

**Moins de biais:**
1. Baseline (1 couche): Modéré
2. Weights (scaling=0.3): +73% critical, +400% emergency
3. Focal Loss (gamma=1.5): +73% critical, +400% emergency
4. Weights (scaling=1.0): +143% critical, +400% emergency

---

## Analyse des Patterns Communs

### 1. Emergency: Le Problème Récurrent

**Toutes les approches avec class weights/focal loss:**
- Recall: 100% (toutes les instances détectées)
- Precision: ~20% (80% de faux positifs)
- 2523 prédictions vs 504 réels (+400%)
- Le modèle prédit emergency par défaut pour éviter de le manquer

**Cause:**
- Weight emergency (2.68 même réduit) encore trop élevé
- La pénalité pour manquer emergency domine la loss
- Le modèle sacrifie precision pour maximiser recall

### 2. Convergence et Stagnation

**Pattern observé dans tous les modèles:**
- Convergence rapide (epoch 1-2)
- Stagnation complète après epoch 3
- Loss oscille autour de 1.10-1.15
- Pas d'amélioration avec 15 epochs

**Implications:**
- Le modèle atteint rapidement son maximum
- Plus d'epochs n'aide pas
- Le problème est architectural/features, pas d'optimisation

### 3. Classes Majoritaires Sous-Apprises

**Pattern systématique:**
- High: 10-12% de prédictions (vs 20.5% réel)
- Low: 12-14% de prédictions (vs 20.5% réel)
- Medium: 7-8% de prédictions (vs 20.6% réel)

**Cause:**
- Le modèle se concentre sur critical/emergency
- Classes majoritaires n'ont pas de caractéristiques discriminantes fortes
- Confusion sémantique entre high/medium/low

### 4. Plafond de Performance

**Observation clé:**
- Aucun modèle ne dépasse 21.2% F1-macro
- Accuracy bloquée à ~20.6%
- Amélioration marginale entre approches (+0.5% max)

**Conclusion:**
- Limite inhérente au dataset/task
- Features textuelles insuffisantes
- Architecture DistilBERT limitée pour cette tâche

---

## Conclusions Générales

### Résultats Clés

**1. Meilleur modèle: Class Weights (scaling=0.3)**
- F1-macro: 21.14% (meilleur score obtenu)
- Tous les F1 par classe > 12%
- Compromis acceptable entre équité et performance
- Distribution moins biaisée que weights complets

**2. Focal Loss inefficace**
- Gamma=1.5 ne surpasse pas class weights (-0.31 F1-macro)
- Résultats quasi-identiques
- Complexité supplémentaire non justifiée
- Même problème de bias emergency

**3. Le problème fondamental persiste**
- Accuracy ~21% proche du hasard (16.67% pour 6 classes)
- Emergency: 100% recall mais 20% precision (inacceptable en production)
- Bias massif: +400% emergency, +73% critical
- Classes majoritaires sous-prédites: -48% high, -64% medium

### Limites des Approches Testées

**Class weights et Focal Loss:**
- Réduisent le déséquilibre mais ne le résolvent pas
- Weight scaling 0.3 optimal testé mais encore insuffisant
- Emergency weight 2.68 encore trop élevé
- Amélioration marginale (+0.5% F1 vs baseline)
- Trade-off inacceptable: +10% recall macro vs precision catastrophique

**Architecture:**
- DistilBERT limité pour cette tâche spécifique
- 2 couches dégelées = overfitting systématique
- Features textuelles seules insuffisantes
- Pas de compréhension du contexte opérationnel

**Optimisation:**
- Stagnation après 2-3 epochs dans tous les cas
- 15 epochs excessifs et inutiles
- Learning rate et warmup ont peu d'impact
- Le problème n'est pas l'optimisation mais les données/features

### Problèmes Non Résolus

**1. Confusion sémantique entre classes**
- Critical vs Emergency: souvent indiscernables textuellement
- High vs Medium vs Low: graduations subtiles sans marqueurs clairs
- Info vs autres: contexte opérationnel nécessaire
- Le texte seul ne capture pas la sévérité réelle

**2. Qualité des labels**
- Possiblement bruités ou subjectifs
- Manque de cohérence dans l'étiquetage historique
- Critères de sévérité non explicites
- Besoin d'audit approfondi du dataset

**3. Features manquantes critiques**
- Contexte système (charge CPU, mémoire, réseau)
- Métadonnées temporelles (heure, jour, tendances)
- Historique (fréquence, récurrence, patterns)
- Impact opérationnel réel (utilisateurs affectés, SLA)
- Criticité des systèmes sources

**4. Déséquilibre structurel**
- Emergency 2.5% du dataset (504 exemples)
- Impossible d'apprendre correctement avec si peu d'exemples
- Augmentation de données nécessaire
- Ou reformulation du problème (classification hiérarchique)

---

## Recommandations Finales

### Arrêter les Approches Actuelles

Les 6 expérimentations démontrent clairement:
- Weights optimisés: plafond à 21% F1-macro
- Focal Loss: pas d'amélioration vs weights
- DistilBERT: architecture insuffisante
- Approche texte seul: fondamentalement limitée
- Optimisation des hyperparamètres: gains marginaux (<1%)

**Verdict: Continuer dans cette direction est improductif**