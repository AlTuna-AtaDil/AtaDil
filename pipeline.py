# pipeline.py
import pickle
import os
import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

# -------------------- 1. Varyant Tanıma Modelini Yükle --------------------
with open("C:/Users/EMRULLAH/PycharmProjects/pythonProject18/models/variant_classifier.pkl", "rb") as f:
    variant_model, vectorizer = pickle.load(f)

# -------------------- 2. Osmanlıca & Ağız Çeviri Modelini Yükle (mt5) --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mt5_path = "models/mt5_translation"  # mt5 modeli bu klasörde olmalı
if os.path.exists(mt5_path):
    mt5_tokenizer = MT5Tokenizer.from_pretrained(mt5_path)
    mt5_model = MT5ForConditionalGeneration.from_pretrained(mt5_path).to(device)
else:
    mt5_tokenizer = None
    mt5_model = None
    print("[Uyarı] mt5 çeviri modeli bulunamadı, placeholder kullanılacak.")


def translate_variant(text):
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
idiom_path = "C:/Users/EMRULLAH/PycharmProjects/pythonProject18/models/idiom_translator.pkl"  # Senin deyim çeviri modelin
if os.path.exists(idiom_path):
    with open(idiom_path, "rb") as f:
        idiom_model = pickle.load(f)
else:
    idiom_model = None
    print("[Uyarı] idiom çeviri modeli bulunamadı, placeholder kullanılacak.")


# pipeline.py (ilgili kısım)

def try_translate_idiom(text, threshold=0.65):
    """
    Deyim modülünü önce dener:
    - Eşik üstü benzerlik yakalarsa (ör. 0.65), deyim olarak kabul eder.
    - Aksi halde None döner ve varyant sınıflandırıcıya geçilir.
    """
    if idiom_model is None:
        return None, 0.0
    out, score = idiom_model.translate(text, return_score=True)
    if out and (not out.startswith("[Eşleşme")) and score >= threshold:
        return out, score
    return None, score


# -------------------- 4. Kullanıcıdan Girdi Al & İşle --------------------
while True:
    text = input("\nMetin girin (çıkmak için q): ").strip()
    if text.lower() == "q":
        break

    # 4.a Önce deyim kontrolü
    idiom_out, score = try_translate_idiom(text, threshold=0.65)
    if idiom_out:
        predicted_label = "deyim"
        translation = idiom_out
    else:
        # 4.b Varyant tahmini
        X = vectorizer.transform([text])
        predicted_label = variant_model.predict(X)[0]

        # 4.c Tahmine göre çeviri
        if predicted_label in ["osmanlica", "Ağız"]:
            translation = translate_variant(text)
        else:
            translation = "[Çeviri modülü bulunamadı]"

    print(f"\n📌 Tahmin Edilen Varyant: {predicted_label}")
    print(f"💬 Çeviri: {translation}")
