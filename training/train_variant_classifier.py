# training/train_variant_classifier.py
import os
import sys
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Üst klasörden modül almak için yol ekleme
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.clean_text import clean_text

os.makedirs("models", exist_ok=True)

# Sadece ağız ve osmanlıca verileri
data_files = [
    "data/osmanlica.csv",
    "data/agiz.csv"
]

dfs = []
for file in data_files:
    if os.path.exists(file) and os.path.getsize(file) > 0:
        try:
            df = pd.read_csv(file)
            if 'text' not in df.columns or 'label' not in df.columns:
                print(f"[Uyarı] {file} dosyasında 'text' ve 'label' başlıkları olmalı!")
                continue
            dfs.append(df)
            print(f"[Bilgi] {file} yüklendi — satır sayısı: {len(df)}")
        except pd.errors.EmptyDataError:
            print(f"[Bilgi] {file} boş, atlanıyor.")
    else:
        print(f"[Bilgi] {file} bulunamadı veya boş.")

if not dfs:
    raise ValueError("Eğitim için kullanılacak hiç veri bulunamadı!")

# Tüm verileri birleştir
df = pd.concat(dfs, ignore_index=True)

# Boş satırları temizle
df = df.dropna(subset=['text', 'label'])
df['label'] = df['label'].astype(str).str.strip().str.lower()
df['text'] = df['text'].astype(str).str.strip()

# Temizlenmiş metin sütunu
df['clean_text'] = df['text'].apply(clean_text)
df = df[df['clean_text'] != ""]

# Sadece 'osmanlica' ve 'ağız' etiketleri
df = df[df['label'].isin(['osmanlica', 'ağız'])]

# Etiket kontrolü
unique_labels = df['label'].unique()
print(f"[Bilgi] Bulunan etiketler: {unique_labels}")
if len(unique_labels) < 2:
    raise ValueError("En az 2 farklı etiket olmalı!")

# TF-IDF ve model eğitimi
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Eğitim / test bölme
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Tahmin ve rapor
y_pred = model.predict(X_test)
print("\nSınıflandırma Sonuçları:\n")
print(classification_report(y_test, y_pred))

# Modeli kaydet
with open("models/variant_classifier.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("\n[+] Model kaydedildi: models/variant_classifier.pkl")
