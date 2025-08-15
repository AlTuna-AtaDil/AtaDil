import pandas as pd

# Dosya yolları (kendine göre değiştir)
csv_osm = "C:/Users/EMRULLAH/PycharmProjects/pythonProject18/data/osmanlica.csv"
csv_agz = "C:/Users/EMRULLAH/PycharmProjects/pythonProject18/data/agiz.csv"

# CSV'leri oku
df_osm = pd.read_csv(csv_osm, encoding="utf-8")
df_agz = pd.read_csv(csv_agz, encoding="utf-8")

# Birleştir
df_all = pd.concat([df_osm, df_agz], ignore_index=True)

# Gereksiz boş satırları temizle
df_all = df_all.dropna(subset=["text","label","standard"]).reset_index(drop=True)

# Kontrol
print("[Bilgi] Satır sayısı:", len(df_all))
print("[Bilgi] Etiket dağılımı:\n", df_all["label"].value_counts())

# Kaydet
df_all.to_csv("AllDataset.csv", index=False, encoding="utf-8")
print("[+] AllDataset.csv oluşturuldu")
