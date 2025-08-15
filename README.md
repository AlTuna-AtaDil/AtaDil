
#BilisimVadisi2025  
#TürkiyeAçıkKaynakPlatformu  
#AtaDil  
#TDDI2025  

# AtaDil

Türkçe odaklı metin işleme aracı.  
Uygulamayı *Streamlit* ile çalıştırıyoruz.

## Gereksinimler
- Python 3.9+
- (Opsiyonel) Sanal ortam kullanımı önerilir.

## Kurulum

```bash
# 1) Depoyu klonla
git clone https://github.com/AlTuna-AtaDil/AtaDil.git
cd AtaDil

# 2) Sanal ortam oluştur ve etkinleştir
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3) Bağımlılıkları kur
pip install --upgrade pip
pip install -r requirements.txt

## Çalıştırma
streamlit run app.py

AtaDil/
├─ app.py                  # Uygulamayı başlatan ana dosya
├─ requirements.txt        # Proje bağımlılıkları
├─ training/
├─ data/
├─ models/
├─ utils.py
└─ ...
