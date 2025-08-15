import pandas as pd
import csv
from io import StringIO


def oku_csv_duzelt(path):
    satirlar = []
    with open(path, "r", encoding="utf-8") as f:
        # Python'un csv modülünü kullan - tırnakları doğru şekilde işler
        csv_reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in csv_reader:
            if len(row) == 0:  # Boş satırları atla
                continue

            # 3 sütuna zorla - eksik varsa boş string ekle, fazla varsa kes
            while len(row) < 3:
                row.append("")
            row = row[:3]  # Sadece ilk 3 sütunu al

            satirlar.append(row)

    return pd.DataFrame(satirlar, columns=["text", "label", "standard"])


# İki dataset yolunu ver
csv_osm = "C:/Users/EMRULLAH/PycharmProjects/pythonProject18/data/osmanlica.csv"
csv_agz = "C:/Users/EMRULLAH/PycharmProjects/pythonProject18/data/agiz.csv"

print("Osmanlıca CSV'yi düzelterek okuyorum...")
df_osm = oku_csv_duzelt(csv_osm)
print(f"Osmanlıca shape: {df_osm.shape}")

print("Ağız CSV'yi düzelterek okuyorum...")
df_agz = oku_csv_duzelt(csv_agz)
print(f"Ağız shape: {df_agz.shape}")

# İlk birkaç satırı kontrol et
print("\nOsmanlıca ilk 3 satır:")
print(df_osm.head(3))
print("\nAğız ilk 3 satır:")
print(df_agz.head(3))

# Birleştir
df_all = pd.concat([df_osm, df_agz], ignore_index=True)

# Kaydet
df_all.to_csv("AllDataset.csv", index=False, encoding="utf-8")

print(f"\n✅ Birleştirme tamamlandı! Toplam satır: {len(df_all)}")
print(f"Final DataFrame shape: {df_all.shape}")
print("Sütunlar:", list(df_all.columns))