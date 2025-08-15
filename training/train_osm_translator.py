# training/train_idiom_translator.py
import os
import sys
import pandas as pd

from idiom_translation.osmanlica_translator import train_and_save_osm_translator, load_osm_translator

# Proje kök yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


DATA_PATH = "C:/Users/EMRULLAH/PycharmProjects/pythonProject18/data/osmanlica2.csv"
MODEL_PATH = "C:/Users/EMRULLAH/PycharmProjects/pythonProject18/models"


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"[HATA] {DATA_PATH} bulunamadı!")

    print(f"[Bilgi] {DATA_PATH} yükleniyor...")
    df = pd.read_csv(DATA_PATH)

    if not {"text", "standard"}.issubset(df.columns):
        raise ValueError("CSV dosyasında 'text' ve 'standard' sütunları olmalı!")

    # Eğitim
    print("[Bilgi] Model eğitiliyor...")
    train_and_save_osm_translator(DATA_PATH)

    # Modeli yükle
    model = load_osm_translator()

    # Doğrulama testi
    print("\n[Bilgi] Doğrulama testi yapılıyor...")
    sample_df = df.sample(min(15, len(df)), random_state=42)
    correct = 0

    for _, row in sample_df.iterrows():
        src = row["text"]
        gold = row["standard"]
        pred = model.translate(src)

        is_correct = (gold.strip().lower() == pred.strip().lower())
        if is_correct:
            correct += 1

        print(f"\n- src: {src}")
        print(f"  gold: {gold}")
        print(f"  pred: {pred}  {'✓' if is_correct else '✗'}")

    acc = (correct / len(sample_df)) * 100
    print(f"\n[Val] Hit@1 doğruluk: {acc:.2f}%  (n={len(sample_df)})")

    print(f"\n[+] Model kaydedildi: {os.path.join(MODEL_PATH, 'osm_translator.pkl')}")


if __name__ == "__main__":
    main()
