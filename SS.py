import streamlit as st
import pandas as pd
import os

# Streamlit sayfa ayarları
st.set_page_config(page_title="AtaDil Veri Etiketleyici", layout="centered")
st.title("📜 AtaDil - Veri Etiketleme Arayüzü")

# Etiket seçenekleri
label_options = ["Ağız", "Osmanlıca", "Sosyal", "Deyim/Atasözü"]

# Girdi alanı
text_input = st.text_area("Etiketlenecek Cümleyi Yazın:", "")
label_selected = st.selectbox("Etiket Seçin:", label_options)

# CSV'yi kaydedeceğimiz yol
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "etiketli_veri.csv")

# Kayıt işlemi
if st.button("✅ Kaydet"):
    if not text_input.strip():
        st.warning("Lütfen bir cümle girin.")
    else:
        new_row = {"text": text_input.strip(), "label": label_selected}

        # Eğer klasör yoksa oluştur
        os.makedirs(DATA_DIR, exist_ok=True)

        # Eğer CSV varsa ekle, yoksa oluştur
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

        df.to_csv(CSV_PATH, index=False)
        st.success("✔️ Veri başarıyla kaydedildi!")

# Ön izleme
if os.path.exists(CSV_PATH):
    st.markdown("---")
    st.subheader("📄 Kaydedilmiş Veriler")
    df = pd.read_csv(CSV_PATH)
    st.dataframe(df.tail(10), use_container_width=True)
