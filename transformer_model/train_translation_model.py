# train_translation_model.py
# mT5 modelini kullanarak Osmanlıca ve Ağız varyant metinlerini
# Standart Türkçeye çeviren modelin eğitimi.

import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from torch.optim import AdamW

# 1. GPU kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Bilgi] Eğitim cihazı: {device}")

# 2. Kullanılacak veri dosyaları (deyim ve sosyal eklenmedi)
data_files = [
    "data/osmanlica.csv",
    "data/agiz.csv"
]

dfs = []
for file in data_files:
    if os.path.exists(file) and os.path.getsize(file) > 0:
        try:
            df = pd.read_csv(file)
            if not {"text", "standard"}.issubset(df.columns):
                raise ValueError(f"[HATA] {file} dosyasında 'text' ve 'standard' sütunları olmalı!")
            # Boş satırları at
            df.dropna(subset=["text", "standard"], inplace=True)
            df["text"] = df["text"].astype(str).str.strip()
            df["standard"] = df["standard"].astype(str).str.strip()
            dfs.append(df)
            print(f"[Bilgi] {file} yüklendi — {len(df)} satır")
        except Exception as e:
            print(f"[HATA] {file} okunamadı: {e}")
    else:
        print(f"[Uyarı] {file} bulunamadı veya boş.")

if not dfs:
    raise ValueError("[HATA] Hiç veri yüklenmedi! Lütfen CSV dosyalarını kontrol et.")

# 3. Tüm verileri birleştir
full_df = pd.concat(dfs, ignore_index=True)
print(f"[Bilgi] Toplam veri satırı: {len(full_df)}")

# 4. Train / Validation ayrımı
train_texts, val_texts, train_labels, val_labels = train_test_split(
    full_df["text"].tolist(),
    full_df["standard"].tolist(),
    test_size=0.2,
    random_state=42
)

# 5. Tokenizer ve model yükleme
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small").to(device)

# 6. Dataset tanımı
class TranslationDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_len=128):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_enc = self.tokenizer(
            "translate Turkish to Turkish: " + self.inputs[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        target_enc = self.tokenizer(
            self.targets[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": input_enc.input_ids.squeeze(),
            "attention_mask": input_enc.attention_mask.squeeze(),
            "labels": target_enc.input_ids.squeeze()
        }

# 7. DataLoader
train_dataset = TranslationDataset(train_texts, train_labels, tokenizer)
val_dataset = TranslationDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# 8. Eğitim
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 3

model.train()
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    epoch_loss = 0
    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()

    print(f"\nLoss: {epoch_loss / len(train_loader):.4f}")

# 9. Modeli kaydet
os.makedirs("models", exist_ok=True)
model.save_pretrained("models/mt5_translation")
tokenizer.save_pretrained("models/mt5_translation")

print("\n[+] Model kaydedildi: models/mt5_translation")
