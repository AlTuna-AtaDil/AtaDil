# app.py
import os
import re
import pickle
import torch
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =========================
# Konfig (yollar)
# =========================
DATASET_PATH = r"C:\Users\EMRULLAH\PycharmProjects\pythonProject18\data\AllDataset.csv"
VARIANT_MODEL_PATH = r"C:\Users\EMRULLAH\PycharmProjects\pythonProject18\models\variant_classifier.pkl"
MT5_MODEL_PATH = r"C:\Users\EMRULLAH\PycharmProjects\pythonProject18\models\teknodl_models\mt5_translation"
IDIOM_MODEL_PATH = r"C:\Users\EMRULLAH\PycharmProjects\pythonProject18\models\idiom_translator.pkl"
OSM_MODEL_PATH = r"C:\Users\EMRULLAH\PycharmProjects\pythonProject18\models\osmanlica_translator.pkl"
AGIZ_MODEL_PATH = r"C:\Users\EMRULLAH\PycharmProjects\pythonProject18\models\agiz_translator.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Cache’li yükleyiciler
# =========================
@st.cache_data(show_spinner=False)
def load_lookup(path):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, encoding="utf-8")
    df["text"] = df["text"].astype(str).str.strip().str.lower()
    df["standard"] = df["standard"].astype(str).str.strip()
    return dict(zip(df["text"], df["standard"]))

@st.cache_resource(show_spinner=False)
def load_router(path):
    with open(path, "rb") as f:
        model, vec = pickle.load(f)
    return model, vec

@st.cache_resource(show_spinner=False)
def load_mt5(path):
    if not os.path.exists(path):
        return None, None
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)
    mdl.eval()
    return tok, mdl

@st.cache_resource(show_spinner=False)
def load_pickle_model(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

lookup_dict = load_lookup(DATASET_PATH)
variant_model, vectorizer = load_router(VARIANT_MODEL_PATH)
tok, mt5_model = load_mt5(MT5_MODEL_PATH)
idiom_model = load_pickle_model(IDIOM_MODEL_PATH)
osm_model = load_pickle_model(OSM_MODEL_PATH)
agiz_model = load_pickle_model(AGIZ_MODEL_PATH)

# =========================
# Yardımcılar
# =========================
def norm(lbl: str) -> str:
    return str(lbl).strip().lower()

def hybrid_translate(text, lookup_dict, pkl_model, mt5_tok, mt5_model, sekme_label, threshold=0.70):
    warnings = []
    translation = None

    # 1) pkl benzerlik modeli
    if pkl_model:
        out, score = pkl_model.translate(text, threshold=threshold, return_score=True)
        if out and not out.startswith("[Eşleşme"):
            translation = out

    # 2) Dataset lookup
    t_clean = text.strip().lower()
    if t_clean in lookup_dict:
        translation = lookup_dict[t_clean]

    # 3) mT5 fallback
    if translation is None:
        if mt5_model and mt5_tok:
            inputs = mt5_tok("translate Turkish to Turkish: " + text,
                             return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                out_ids = mt5_model.generate(**inputs, max_length=128, num_beams=4,
                                             no_repeat_ngram_size=3, length_penalty=1.0, early_stopping=True)
            translation = mt5_tok.decode(out_ids[0], skip_special_tokens=True).strip()
        else:
            translation = f"[ST Çeviri placeholder] {text}"

    # 4) Classifier uyarısı
    pred = variant_model.predict(vectorizer.transform([text]))[0]
    if norm(pred) != norm(sekme_label):
        warnings.append(f"⚠ Bu cümle **{pred}** özelliğine daha yakın görünüyor.")

    return translation, warnings

def try_translate_idiom(text: str, threshold: float = 0.65):
    if idiom_model is None:
        return None, 0.0
    out, score = idiom_model.translate(text, return_score=True)
    if out and not out.startswith("[Eşleşme") and score >= threshold:
        return out, score
    return None, score

# =========================
# UI
# =========================
st.set_page_config(page_title="AtaDil Demo", page_icon="📜", layout="centered")

if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    st.title("📜 AtaDil Projesi")
    st.write("Osmanlıca, Ağız ve Deyim Tanıma & Çeviri Sistemi")
    cols = st.columns(3)
    if cols[1].button("Başla 🚀", use_container_width=True):
        st.session_state.page = "main"
    if not lookup_dict:
        st.warning("Dataset lookup yüklenemedi (AllDataset.csv bulunamadı).")
    if tok is None or mt5_model is None:
        st.info("mT5 modeli bulunamadı — sadece lookup/pkl çalışacak.")
    if idiom_model is None:
        st.info("Deyim modeli bulunamadı — deyim sekmesi kısıtlı çalışır.")
    st.stop()

st.title("📚 AtaDil Çeviri ve Tanıma")
tabs = st.tabs(["Osmanlıca", "Ağız", "Deyim"])

# -------- Osmanlıca --------
with tabs[0]:
    st.subheader("Osmanlıca → Standart Türkçe")
    with st.form("osm_form", clear_on_submit=False):
        osm_text = st.text_area("Cümleyi giriniz", height=120)
        submit = st.form_submit_button("Çevir (Osmanlıca)")
    if submit and osm_text.strip():
        tr, warns = hybrid_translate(osm_text, lookup_dict, osm_model, tok, mt5_model, "osmanlica")
        st.success("📌 Tür: **Osmanlıca**")
        st.write("**💬 Standart Türkçe:**")
        st.write(tr)
        for w in warns:
            st.warning(w)

# -------- Ağız --------
with tabs[1]:
    st.subheader("Ağız → Standart Türkçe")
    with st.form("agiz_form", clear_on_submit=False):
        agiz_text = st.text_area("Cümleyi giriniz", height=120)
        submit = st.form_submit_button("Çevir (Ağız)")
    if submit and agiz_text.strip():
        tr, warns = hybrid_translate(agiz_text, lookup_dict, agiz_model, tok, mt5_model, "ağız")
        st.success("📌 Tür: **Ağız**")
        st.write("**💬 Standart Türkçe:**")
        st.write(tr)
        for w in warns:
            st.warning(w)

# -------- Deyim --------
with tabs[2]:
    st.subheader("İngilizce Deyim → Türkçe")
    with st.form("deyim_form", clear_on_submit=False):
        deyim_text = st.text_area("İngilizce deyimi giriniz", placeholder="break a leg, bite the bullet ...",
                                  height=120)
        submit = st.form_submit_button("Çevir (Deyim)")
    if submit and deyim_text.strip():
        if not re.fullmatch(r"[A-Za-z\s\-'’]+", deyim_text.strip()):
            st.error("⚠ Lütfen **sadece İngilizce deyimler** giriniz.")
        else:
            out, sc = try_translate_idiom(deyim_text)
            if out:
                st.success("📌 Tür: **Deyim**")
                st.write("**💬 Türkçe Karşılığı:**")
                st.write(out)
            else:
                st.error("Bu ifade **deyim** değil.")
