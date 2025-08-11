import re

def clean_text(text):
    """
    Metni temizler:
    - Küçük harfe çevirir
    - Noktalama işaretlerini kaldırır
    - Fazla boşlukları temizler
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini sil
    text = re.sub(r'\s+', ' ', text).strip()  # Fazla boşlukları tek boşluğa indir
    return text



