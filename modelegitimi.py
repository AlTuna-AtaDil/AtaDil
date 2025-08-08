#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEKNOFEST 2025 - Model Eğitimi Script
ACİL 11 GÜNLÜK PLAN - GÜN 3-4

1. ADIM: Varyant Tanıma Modeli (modern/osmanlica/sosyal/agiz)
2. ADIM: BERTurk ile Fine-tuning
3. ADIM: Model Değerlendirme
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, pipeline
)
import torch
from torch.utils.data import Dataset
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class TurkishVariantDataset(Dataset):
    """Türkçe varyant dataset'i için özel Dataset sınıfı"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class TurkishVariantClassifier:
    """Türkçe Varyant Tanıma Modeli"""

    def __init__(self):
        self.model_name = "dbmdz/bert-base-turkish-cased"  # BERTurk
        self.tokenizer = None
        self.model = None
        self.label_to_id = {'modern': 0, 'osmanlica': 1, 'sosyal': 2, 'agiz': 3}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def load_data(self, csv_file='turkish_variant_dataset.csv'):
        """CSV'den veriyi yükle"""
        print("📊 Dataset yükleniyor...")

        df = pd.read_csv(csv_file)
        print(f"Toplam örnek sayısı: {len(df)}")
        print("\nKategori dağılımı:")
        print(df['etiket'].value_counts())

        # Etiketleri sayısal değerlere çevir
        df['label_id'] = df['etiket'].map(self.label_to_id)

        return df

    def prepare_data(self, df, test_size=0.2):
        """Veriyi train/test olarak ayır"""
        print("🔄 Veri hazırlanıyor...")

        texts = df['metin'].tolist()
        labels = df['label_id'].tolist()

        # Train-test split
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )

        print(f"Eğitim verisi: {len(train_texts)}")
        print(f"Test verisi: {len(test_texts)}")

        return train_texts, test_texts, train_labels, test_labels

    def initialize_model(self):
        """Model ve tokenizer'ı başlat"""
        print("🤖 Model yükleniyor (BERTurk)...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=4  # 4 kategori: modern, osmanlica, sosyal, agiz
        )

        print("✅ Model başarıyla yüklendi!")

    def create_datasets(self, train_texts, test_texts, train_labels, test_labels):
        """PyTorch Dataset'leri oluştur"""
        train_dataset = TurkishVariantDataset(
            train_texts, train_labels, self.tokenizer
        )
        test_dataset = TurkishVariantDataset(
            test_texts, test_labels, self.tokenizer
        )

        return train_dataset, test_dataset

    def train_model(self, train_dataset, test_dataset):
        """Modeli eğit"""
        print("🚀 Model eğitimi başlıyor...")

        # Eğitim parametreleri (HIZLI EĞİTİM İÇİN)
        training_args = TrainingArguments(
            output_dir='./turkish_variant_model',
            num_train_epochs=3,  # Kısa süre için 3 epoch
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        # Trainer oluştur
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        # Eğitimi başlat
        print("⏳ Eğitim devam ediyor... (Bu 10-15 dakika sürebilir)")
        trainer.train()

        # Modeli kaydet
        trainer.save_model('./turkish_variant_model')
        self.tokenizer.save_pretrained('./turkish_variant_model')

        print("✅ Model eğitimi tamamlandı ve kaydedildi!")

        return trainer

    def evaluate_model(self, trainer, test_texts, test_labels):
        """Modeli değerlendir"""
        print("📈 Model değerlendiriliyor...")

        # Tahmin yap
        predictions = trainer.predict(trainer.eval_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)

        # Accuracy hesapla
        accuracy = accuracy_score(test_labels, y_pred)
        print(f"🎯 Test Accuracy: {accuracy:.4f}")

        # Classification report
        print("\n📊 Detaylı Performans Raporu:")
        target_names = ['modern', 'osmanlica', 'sosyal', 'agiz']
        print(classification_report(test_labels, y_pred, target_names=target_names))

        # Confusion matrix
        cm = confusion_matrix(test_labels, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Gerçek Etiket')
        plt.xlabel('Tahmin Edilen Etiket')
        plt.savefig('confusion_matrix.png')
        plt.show()

        return accuracy, y_pred

    def create_pipeline(self):
        """Kolay kullanım için pipeline oluştur"""
        print("🔧 Pipeline oluşturuluyor...")

        classifier = pipeline(
            "text-classification",
            model="./turkish_variant_model",
            tokenizer="./turkish_variant_model"
        )

        return classifier

    def test_examples(self, classifier):
        """Test örnekleri ile modeli dene"""
        print("🧪 Test örnekleri ile model deneniyor...")

        test_examples = [
            "Bugün hava çok güzel, dışarı çıkalım.",  # modern
            "Bu gün hava pek hoş idi, gezintiye çıktık.",  # osmanlica
            "Abi bu film baya iyi ya, kesin izle.",  # sosyal
            "Ava çok güzel olmuş bea, çıkalım dışarı."  # agiz
        ]

        print("\n🔍 Tahmin Sonuçları:")
        print("-" * 50)

        for text in test_examples:
            result = classifier(text)
            predicted_label = result[0]['label']
            confidence = result[0]['score']

            print(f"Metin: {text}")
            print(f"Tahmin: {predicted_label} (Güven: {confidence:.3f})")
            print("-" * 50)

    def run_full_training(self, csv_file='turkish_variant_dataset.csv'):
        """Tam eğitim sürecini çalıştır"""
        print("🚀 TÜRKÇE VARYANT TANIMA MODELİ EĞİTİMİ BAŞLIYOR!")
        print("=" * 60)

        # 1. Veri yükle
        df = self.load_data(csv_file)

        # 2. Veriyi hazırla
        train_texts, test_texts, train_labels, test_labels = self.prepare_data(df)

        # 3. Modeli başlat
        self.initialize_model()

        # 4. Dataset'leri oluştur
        train_dataset, test_dataset = self.create_datasets(
            train_texts, test_texts, train_labels, test_labels
        )

        # 5. Modeli eğit
        trainer = self.train_model(train_dataset, test_dataset)

        # 6. Modeli değerlendir
        accuracy, predictions = self.evaluate_model(trainer, test_texts, test_labels)

        # 7. Pipeline oluştur
        classifier = self.create_pipeline()

        # 8. Test örnekleri
        self.test_examples(classifier)

        print(f"\n🎉 MODEL EĞİTİMİ TAMAMLANDI!")
        print(f"📊 Final Accuracy: {accuracy:.4f}")
        print("💾 Model './turkish_variant_model' klasörüne kaydedildi")
        print("\n🔥 SONRAKİ ADIMLAR:")
        print("1. Çeviri modülünü geliştir")
        print("2. Demo arayüzünü hazırla")
        print("3. GitHub'a yükle!")

        return classifier, accuracy


# HEMEN ÇALIŞTIR!
if __name__ == "__main__":
    # Eğitimi başlat
    trainer = TurkishVariantClassifier()

    # CUDA varsa kullan (hızlandırır)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Kullanılan cihaz: {device}")

    # Tam eğitim sürecini çalıştır
    classifier, accuracy = trainer.run_full_training()

    print("✅ Hazır! Şimdi çeviri modülüne geçebilirsin!")