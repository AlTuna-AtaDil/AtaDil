# idiom_translation/idiom_translator.py
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util

MODEL_PATH = "C:/Users/EMRULLAH/PycharmProjects/pythonProject18/models"
AGIZ_MODEL_FILE = os.path.join(MODEL_PATH, "agiz_translator.pkl")


class AgizTranslator:
    def __init__(self, agiz, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        """
        idioms: { "ingilizce_deyim": "türkçe_anlam" }
        """
        self.agiz = agiz
        self.src_texts = list(agiz.keys())
        self.tgt_texts = list(agiz.values())
        self.embedder = SentenceTransformer(model_name)
        self.src_embeddings = self.embedder.encode(self.src_texts, convert_to_tensor=True)

    def translate(self, text, threshold=0.75, return_score=False):
        query_emb = self.embedder.encode(text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_emb, self.src_embeddings)[0]
        best_idx = int(np.argmax(cosine_scores))
        best_score = float(cosine_scores[best_idx])

        if best_score >= threshold:
            result = self.tgt_texts[best_idx]
        else:
            result = "[Eşleşme bulunamadı]"

        if return_score:
            return result, best_score
        else:
            return result


def train_and_save_agiz_translator(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)

    if not {"text", "standard"}.issubset(df.columns):
        raise ValueError("CSV dosyasında 'text' ve 'standard' sütunları olmalı!")

    agiz_dict = dict(zip(df["text"], df["standard"]))
    translator = AgizTranslator(agiz_dict)

    os.makedirs(MODEL_PATH, exist_ok=True)
    with open(AGIZ_MODEL_FILE, "wb") as f:
        pickle.dump(translator, f)

    print(f"[+] Model kaydedildi: {AGIZ_MODEL_FILE}")


def load_agiz_translator():
    if not os.path.exists(AGIZ_MODEL_FILE):
        raise FileNotFoundError(f"[HATA] {AGIZ_MODEL_FILE} bulunamadı! Önce eğitimi çalıştır.")
    with open(AGIZ_MODEL_FILE, "rb") as f:
        return pickle.load(f)


# Doğrudan test etmek istersen
if __name__ == "__main__":
    csv_path = r"C:\Users\EMRULLAH\PycharmProjects\pythonProject18\data\agiz2.csv"  # Burayı kendi deyim CSV dosyana göre ayarla
    train_and_save_agiz_translator(csv_path)

    model = load_agiz_translator()
    print(model.translate("Gene kızayrum kendime o kağıdı niye göstermedum ben"))
