# 🌍 Hausa → Zarma Neural Machine Translation

Projet de traduction automatique des langues locales africaines (Hausa → Zarma) basé sur les Transformers et le fine-tuning de mT5.

---

## 📌 Description

Ce projet vise à développer un système de traduction automatique entre le **Hausa** et le **Zarma**, deux langues largement parlées au Niger et en Afrique de l’Ouest.

Le modèle utilisé est basé sur **mT5 (Multilingual Text-to-Text Transformer)** et entraîné sur un dataset construit à partir d’un dictionnaire linguistique enrichi et augmenté.

---

## 🎯 Objectifs

- Construire un dataset parallèle Hausa ↔ Zarma
- Nettoyer et normaliser les données linguistiques
- Entraîner un modèle de traduction basé sur Transformers
- Évaluer les performances avec des métriques NLP
- Déployer une application interactive avec Streamlit

---

## 🏗️ Architecture du projet


haoussa-zarma-translator/
│
├── data/
│ ├── raw/ # Dictionnaire brut
│ ├── processed/ # Dataset nettoyé et augmenté
│
├── notebooks/
│ ├── 01_exploration.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_training.ipynb
│ ├── 04_evaluation.ipynb
│
├── src/
│ ├── data_loader.py
│ ├── preprocess.py
│ ├── train.py
│ ├── evaluate.py
│ ├── inference.py
│
├── app/
│ └── streamlit_app.py
│
├── models/
│ └── mt5-haoussa-zarma/
│
├── requirements.txt
├── environment.yml
└── README.md


---

## 🧠 Modèle utilisé

- **mT5-small**  
- Framework : Hugging Face Transformers  
- Type : Sequence-to-Sequence (Text-to-Text)

👉 Le modèle apprend à générer une traduction Zarma à partir d’une phrase en Hausa.

---

## 📊 Dataset

Le dataset a été construit à partir d’un dictionnaire bilingue structuré :

- Salutations
- Famille
- Nombres
- Couleurs
- Verbes
- Nourriture
- Santé
- École
- etc.

### 🔧 Préprocessing :
- Nettoyage des textes
- Normalisation (lowercase, trimming)
- Gestion des synonymes
- Augmentation des données (variations de phrases)

---

## ⚙️ Pipeline NLP

### 1. Exploration des données
Analyse du vocabulaire, des longueurs et des catégories.

### 2. Préprocessing
Tokenization avec mT5 tokenizer et création des datasets HuggingFace.

### 3. Fine-tuning
Entraînement du modèle avec Trainer API.

### 4. Évaluation
Métriques utilisées :

- BLEU score
- ROUGE
- chrF (adapté aux langues morphologiquement pauvres)

---

## 📦 Installation

### 1. Créer l’environnement

```bash
conda create -n nlp-haoussa python=3.10
conda activate nlp-haoussa
2. Installer dépendances
pip install -r requirements.txt
📚 Requirements
transformers
datasets
torch
sentencepiece
scikit-learn
pandas
numpy
matplotlib
seaborn
evaluate
sacrebleu
streamlit
🚀 Entraînement du modèle
python src/train.py

Ou via notebook :

notebooks/03_training.ipynb
📊 Évaluation
python src/evaluate.py

Ou notebook :

notebooks/04_evaluation.ipynb
🌐 Application Streamlit
Lancer l’interface :
cd app
streamlit run streamlit_app.py
Fonctionnalités :
Traduction Hausa → Zarma
Interface interactive
Historique des traductions
Exemples rapides
🧪 Exemple d’utilisation
Input :
Sannu
Output :
Fofo
📈 Résultats
BLEU score : baseline modéré (dépend taille dataset)
chrF : plus adapté aux langues locales
Bonne performance sur phrases courtes et fréquentes
⚠️ Limitations
Dataset limité en taille
Performances faibles sur phrases complexes
Nécessite enrichissement des données
🔮 Améliorations futures
Augmentation massive du dataset
Back-translation
Ajout Zarma → Hausa
Déploiement API REST
Fine-tuning sur mT5-large
Support multi-langues africaines
👨‍💻 Auteur

Projet réalisé dans le cadre d’un cours de NLP.

🧠 Conclusion

Ce projet démontre une chaîne complète de NLP :

Données → Préprocessing → Fine-tuning → Évaluation → Déploiement

Il constitue une base solide pour des systèmes de traduction des langues locales africaines.

🌍 Vision

Créer des modèles IA accessibles aux langues sous-représentées en Afrique.