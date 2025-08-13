# idiom_translation/idiom_translator.py
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util

MODEL_PATH = "C:/Users/EMRULLAH/PycharmProjects/pythonProject18/models"
IDIOM_MODEL_FILE = os.path.join(MODEL_PATH, "idiom_translator.pkl")


class IdiomTranslator:
    def __init__(self, idioms, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        """
        idioms: { "ingilizce_deyim": "türkçe_anlam" }
        """
        self.idioms = idioms
        self.src_texts = list(idioms.keys())
        self.tgt_texts = list(idioms.values())
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


def train_and_save_idiom_translator(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)

    if not {"text", "standard"}.issubset(df.columns):
        raise ValueError("CSV dosyasında 'text' ve 'standard' sütunları olmalı!")

    idioms_dict = dict(zip(df["text"], df["standard"]))
    translator = IdiomTranslator(idioms_dict)

    os.makedirs(MODEL_PATH, exist_ok=True)
    with open(IDIOM_MODEL_FILE, "wb") as f:
        pickle.dump(translator, f)

    print(f"[+] Model kaydedildi: {IDIOM_MODEL_FILE}")


def load_idiom_translator():
    if not os.path.exists(IDIOM_MODEL_FILE):
        raise FileNotFoundError(f"[HATA] {IDIOM_MODEL_FILE} bulunamadı! Önce eğitimi çalıştır.")
    with open(IDIOM_MODEL_FILE, "rb") as f:
        return pickle.load(f)


# Doğrudan test etmek istersen
if __name__ == "__main__":
    csv_path = "data/deyim.csv"  # Burayı kendi deyim CSV dosyana göre ayarla
    train_and_save_idiom_translator(csv_path)

    model = load_idiom_translator()
    print(model.translate("Once in a blue moon"))
