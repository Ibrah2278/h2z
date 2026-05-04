import streamlit as st
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# =========================
# CONFIG
# =========================
MODEL_PATH = "../models/mt5-haoussa-zarma"

st.set_page_config(
    page_title="Hausa → Zarma Translator",
    page_icon="🌍",
    layout="centered"
)

# =========================
# LOAD MODEL (cached)
# =========================
@st.cache_resource
def load_model():
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    tokenizer = MT5Tokenizer.from_pretrained(MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, tokenizer, device


model, tokenizer, device = load_model()

# =========================
# TRANSLATION FUNCTION
# =========================
def translate(text):
    input_text = "translate Hausa to Zarma: " + text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=32,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =========================
# UI
# =========================

st.title("🌍 Traducteur Hausa → Zarma")
st.markdown("Un modèle NLP basé sur Transformers pour traduire des langues locales.")

# =========================
# INPUT
# =========================
user_input = st.text_area(
    "✍️ Entrez votre texte en Haoussa :",
    placeholder="Ex: Sannu, Ina gidan ku?"
)

# =========================
# BUTTON
# =========================
if st.button("🔄 Traduire"):

    if user_input.strip() == "":
        st.warning("Veuillez entrer un texte.")
    else:
        with st.spinner("Traduction en cours..."):
            result = translate(user_input)

        st.success("Traduction :")
        st.markdown(f"### 🗣️ {result}")

        # =========================
        # HISTORY
        # =========================
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append((user_input, result))


# =========================
# HISTORY DISPLAY
# =========================
if "history" in st.session_state and len(st.session_state.history) > 0:
    st.markdown("## 📜 Historique")

    for i, (src, tgt) in enumerate(reversed(st.session_state.history[-5:])):
        st.markdown(f"**Hausa:** {src}")
        st.markdown(f"**Zarma:** {tgt}")
        st.markdown("---")


# =========================
# EXAMPLES
# =========================
st.markdown("## ⚡ Exemples rapides")

examples = [
    "Sannu",
    "Nagode",
    "Ina gidan ku?",
    "Ina jin yunwa",
    "Ina zuwa kasuwa"
]

cols = st.columns(len(examples))

for i, example in enumerate(examples):
    if cols[i].button(example):
        result = translate(example)
        st.markdown(f"**Résultat:** {result}")


# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "Projet NLP – Traduction Hausa → Zarma | Basé sur mT5"
)