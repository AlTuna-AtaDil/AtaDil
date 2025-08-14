# pipeline.py

import os
import pickle
import torch
import pandas as pd
import re
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

# -------------------- 0. Dataset Lookup --------------------
DATASET_PATH = r"C:\Users\EMRULLAH\PycharmProjects\pythonProject18\data\AllDataset.csv"

def normalize_text(s):
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\sçğıöşü]", "", s)  # noktalama temizle
    return s

if os.path.exists(DATASET_PATH):
    df_lookup = pd.read_csv(DATASET_PATH, encoding="utf-8")
    df_lookup["text"] = df_lookup["text"].apply(normalize_text)
    df_lookup["standard"] = df_lookup["standard"].astype(str).str.strip()
    lookup_dict = dict(zip(df_lookup["text"], df_lookup["standard"]))
    print(f"[+] Lookup sözlüğü yüklendi. Toplam {len(lookup_dict)} giriş.")
else:
    lookup_dict = {}
    print("[Uyarı] Dataset bulunamadı, lookup devre dışı.")

# -------------------- 1. Varyant Tanıma Modelini Yükle --------------------
VARIANT_MODEL_PATH = r"C:\Users\EMRULLAH\PycharmProjects\pythonProject18\models\variant_classifier.pkl"

with open(VARIANT_MODEL_PATH, "rb") as f:
    variant_model, vectorizer = pickle.load(f)

print("[+] Varyant tanıma modeli yüklendi.")

# -------------------- 2. Osmanlıca & Ağız Çeviri Modelini Yükle (mT5) --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MT5_MODEL_PATH = r"C:\Users\EMRULLAH\PycharmProjects\pythonProject18\models\teknodl_models\mt5_translation"

if os.path.exists(MT5_MODEL_PATH):
    mt5_tokenizer = MT5Tokenizer.from_pretrained(MT5_MODEL_PATH)
    mt5_model = MT5ForConditionalGeneration.from_pretrained(MT5_MODEL_PATH).to(device)
    print("[+] mT5 çeviri modeli yüklendi.")
else:
    mt5_tokenizer = None
    mt5_model = None
    print("[Uyarı] mT5 çeviri modeli bulunamadı, sadece lookup kullanılacak.")

def translate_variant(text):
    text_clean = normalize_text(text)

    # 1️⃣ Önce dataset lookup
    if text_clean in lookup_dict:
        return lookup_dict[text_clean]

    # 2️⃣ Fallback olarak mT5 kullan
    if mt5_model is None:
        return f"[ST Çeviri placeholder] {text}"

    inputs = mt5_tokenizer(
        "translate Turkish to Turkish: " + text,
        return_tensors="pt",
        max_length=128,
        truncation=True
    ).to(device)

    outputs = mt5_model.generate(**inputs, max_length=128)
    return mt5_tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------- 3. Deyim Çeviri Modülünü Yükle --------------------
IDIOM_MODEL_PATH = r"C:\Users\EMRULLAH\PycharmProjects\pythonProject18\models\idiom_translator.pkl"

if os.path.exists(IDIOM_MODEL_PATH):
    with open(IDIOM_MODEL_PATH, "rb") as f:
        idiom_model = pickle.load(f)
else:
    idiom_model = None
    print("[Uyarı] Deyim çeviri modeli bulunamadı, placeholder kullanılacak.")

def try_translate_idiom(text, threshold=0.65):
    if idiom_model is None:
        return None, 0.0

    out, score = idiom_model.translate(text, return_score=True)
    if out and not out.startswith("[Eşleşme") and score >= threshold:
        return out, score
    return None, score

# -------------------- 4. Kullanıcıdan Girdi Al & İşle --------------------
def main():
    while True:
        text = input("\nMetin girin (çıkmak için q): ").strip()
        if text.lower() == "q":
            break

        # 4.a Deyim kontrolü
        idiom_out, score = try_translate_idiom(text)
        if idiom_out:
            predicted_label = "deyim"
            translation = idiom_out
        else:
            # 4.b Varyant tahmini
            X = vectorizer.transform([text])
            predicted_label = variant_model.predict(X)[0]

            # 4.c Çeviri (lookup + mT5 fallback)
            if predicted_label.lower() in ["osmanlica", "ağız"]:
                translation = translate_variant(text)
            else:
                translation = "[Çeviri modülü bulunamadı]"

        print(f"\n📌 Tahmin Edilen Varyant: {predicted_label}")
        print(f"💬 Çeviri: {translation}")

if __name__ == "__main__":
    main()
