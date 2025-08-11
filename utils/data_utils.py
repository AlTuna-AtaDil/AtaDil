# utils/data_utils.py
import pandas as pd
from preprocessing.clean_text import clean_text


def load_dataset(file_path, clean=True):
    """
    CSV dosyasını yükler ve istenirse temizler.

    Args:
        file_path (str): Dosya yolu
        clean (bool): Metinler temizlensin mi?

    Returns:
        pandas.DataFrame: Temizlenmiş veri çerçevesi
    """
    df = pd.read_csv(file_path)

    # Zorunlu sütunları kontrol et
    required_columns = {'text', 'label', 'standard'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Veri dosyasında şu sütunlar eksik: {required_columns - set(df.columns)}")

    if clean:
        df['text'] = df['text'].apply(clean_text)
        df['standard'] = df['standard'].apply(clean_text)

    return df
