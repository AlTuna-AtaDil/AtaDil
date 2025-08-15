import pandas as pd
import numpy as np
import os, pickle, random, gc, torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
CSV_PATH = r"C:\Users\EMRULLAH\PycharmProjects\pythonProject18\data\AllDataset.csv"  # kendi yolun
SAVE_DIR = r"C:\Users\EMRULLAH\PycharmProjects\pythonProject18\models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================
# 2. Veri Yükleme & Temizleme
# ======================
df = pd.read_csv(CSV_PATH, encoding="utf-8")

# Sütun kontrolü
assert set(["text","label","standard"]).issubset(df.columns)

# Etiket normalize (SADECE Osmanlıca ve Ağız)
def norm_label(x):
    x = str(x).strip()
    # Tırnak işaretlerini temizle
    x = x.replace('"', '').replace('"', '').replace('"', '').replace("'", '')
    x = x.lower()

    if x in ["ağız","agiz","agız","şive"]:
        return "Ağız"
    if x in ["osmanlıca","osmanlica"]:
        return "osmanlica"
    return x

df["label"] = df["label"].apply(norm_label)

# Sadece ilgili iki sınıfı tut
df = df[df["label"].isin(["osmanlica","Ağız"])].reset_index(drop=True)

# Çok kısa/uzun metinleri filtrele
def tok_len(s): return len(str(s).split())
df = df[(df["text"].map(tok_len).between(1,256)) & (df["standard"].map(tok_len).between(1,256))]

print("[Bilgi] Etiket dağılımı:\n", df.label.value_counts())

# ======================
# 3. Varyant Tanıma (Router Model)
# ======================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

train_df, val_df = train_test_split(
    df[["text","label"]], test_size=0.2, random_state=SEED, stratify=df["label"]
)

router = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=3)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

router.fit(train_df["text"], train_df["label"])
pred = router.predict(val_df["text"])
print("\n[Router Model Sonuçları]")
print(classification_report(val_df["label"], pred))

with open(f"{SAVE_DIR}/router_variant.pkl","wb") as f:
    pickle.dump(router, f)
print("[+] Router model kaydedildi:", f"{SAVE_DIR}/router_variant.pkl")

# ======================
# 4. mT5 Dönüştürme Modeli
# ======================
def prefix_for_label(lbl):
    if lbl == "osmanlica":
        return "translate ottoman to tr: "
    if lbl == "agiz":
        return "normalize dialect to tr: "
    return ""

# Train/val split
train_df_mt5, val_df_mt5 = train_test_split(
    df, test_size=0.1, random_state=SEED, stratify=df["label"]
)

for split_df in (train_df_mt5, val_df_mt5):
    split_df["input"]  = split_df.apply(lambda r: prefix_for_label(r["label"]) + str(r["text"]), axis=1)
    split_df["target"] = split_df["standard"].astype(str)

ds = DatasetDict({
    "train": Dataset.from_pandas(train_df_mt5[["input","target"]].reset_index(drop=True)),
    "validation": Dataset.from_pandas(val_df_mt5[["input","target"]].reset_index(drop=True)),
})
print(ds)

# Tokenizer & model
model_id = "google/mt5-small"
tok   = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# VRAM dostu ayarlar
model.config.use_cache = False
model.gradient_checkpointing_enable()

max_len = 96
def preprocess(batch):
    mi = tok(batch["input"], max_length=max_len, truncation=True)
    with tok.as_target_tokenizer():
        labels = tok(batch["target"], max_length=max_len, truncation=True)
    mi["labels"] = [[(i if i != tok.pad_token_id else -100) for i in ids] for ids in labels["input_ids"]]
    return mi

ds_tok = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model, pad_to_multiple_of=8)

args = Seq2SeqTrainingArguments(
    output_dir=f"{SAVE_DIR}/mt5_translation",
    learning_rate=5e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    fp16=True,
    optim="adafactor",

    eval_strategy="steps",
    eval_steps=800,
    save_steps=800,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    logging_steps=100,
    report_to="none",
    predict_with_generate=True,
    generation_max_length=96,
    eval_accumulation_steps=16,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=ds_tok["train"],
    eval_dataset=ds_tok["validation"],
    data_collator=collator,
    tokenizer=tok
)

gc.collect()
torch.cuda.empty_cache()
trainer.train()

trainer.save_model(f"{SAVE_DIR}/mt5_translation")
tok.save_pretrained(f"{SAVE_DIR}/mt5_translation")
print("✅ mT5 modeli kaydedildi:", f"{SAVE_DIR}/mt5_translation")

