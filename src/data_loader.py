import pandas as pd
import json
import random
import re
from pathlib import Path

# =========================
# CONFIG
# =========================
INPUT_PATH = "data/raw/dataset.csv"
OUTPUT_PATH = "data/processed/dataset.json"

# =========================
# CLEANING FUNCTIONS
# =========================

def clean_text(text):
    if pd.isna(text):
        return None

    text = str(text).strip().lower()

    # enlever espaces multiples
    text = re.sub(r"\s+", " ", text)

    # enlever caractères bizarres
    text = text.replace("�", "")

    return text


def split_variants(text):
    """
    Split sur / ou virgule si plusieurs variantes
    """
    if text is None:
        return []

    parts = re.split(r"/|,", text)
    return [p.strip() for p in parts if p.strip()]


# =========================
# DATA AUGMENTATION
# =========================

def augment_pair(hausa, zarma):
    """
    Génère variations simples
    """
    augmented = []

    # version originale
    augmented.append((hausa, zarma))

    # variation 1: ajouter ponctuation
    augmented.append((hausa + " ?", zarma))
    augmented.append((hausa + " !", zarma))

    # variation 2: majuscule
    augmented.append((hausa.capitalize(), zarma))

    # variation 3: petit bruit (robustesse)
    if len(hausa.split()) > 1:
        words = hausa.split()
        random.shuffle(words)
        augmented.append((" ".join(words), zarma))

    return augmented


# =========================
# MAIN PROCESS
# =========================

def build_dataset():
    df = pd.read_csv(INPUT_PATH)

    dataset = []

    for _, row in df.iterrows():

        hausa_raw = clean_text(row["haoussa"])
        zarma_raw = clean_text(row["zarma"])

        # skip si vide
        if not hausa_raw or not zarma_raw:
            continue

        # gérer synonymes
        hausa_variants = split_variants(hausa_raw)
        zarma_variants = split_variants(zarma_raw)

        # combinaisons
        for h in hausa_variants:
            for z in zarma_variants:

                augmented_pairs = augment_pair(h, z)

                for h_aug, z_aug in augmented_pairs:
                    dataset.append({
                        "translation": {
                            "hausa": h_aug,
                            "zarma": z_aug
                        }
                    })

    # shuffle dataset
    random.shuffle(dataset)

    print(f"Dataset size: {len(dataset)}")

    # save
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    build_dataset()