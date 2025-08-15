import re

# DOSYA YOLUNU BURAYA YAZIN
dosya_yolu = "C:/Users/EMRULLAH/Downloads/idioms_deyim_eklenmis.txt"  # Bu kısmı kendi dosyanızın adıyla değiştirin


def nokta_kaldir(satir):
    """Satırdan nokta (.) karakterlerini kaldırır"""
    # Tüm nokta karakterlerini kaldır
    yeni_satir = satir.replace('.', '')
    return yeni_satir


def fazla_bosluk_temizle(satir):
    """Fazla boşlukları düzelt"""
    # Çoklu boşlukları tek boşluğa çevir
    satir = re.sub(r'\s+', ' ', satir)
    # Başında ve sonunda boşluk varsa kaldır
    satir = satir.strip()
    return satir


try:
    # Dosyayı oku
    with open(dosya_yolu, 'r', encoding='utf-8') as dosya:
        satirlar = dosya.readlines()

    print(f"Dosya okundu: {len(satirlar)} satır bulundu.")

    yeni_satirlar = []
    degisiklik_sayisi = 0

    for i, satir in enumerate(satirlar):
        orijinal_satir = satir.rstrip('\n')

        # Nokta karakterlerini kaldır
        yeni_satir = nokta_kaldir(orijinal_satir)

        # Fazla boşlukları düzelt
        yeni_satir = fazla_bosluk_temizle(yeni_satir)

        # Değişiklik olup olmadığını kontrol et
        if orijinal_satir != yeni_satir:
            degisiklik_sayisi += 1
            if i < 10:  # İlk 10 değişikliği göster
                print(f"Satır {i + 1}:")
                print(f"  Önce: '{orijinal_satir}'")
                print(f"  Sonra: '{yeni_satir}'")
                print()

        yeni_satirlar.append(yeni_satir)

    # Yeni içeriği dosyaya yaz
    cikis_dosya = dosya_yolu.replace('.txt', '_nokta_kaldirilmis.txt')
    with open(cikis_dosya, 'w', encoding='utf-8') as dosya:
        for satir in yeni_satirlar:
            dosya.write(satir + '\n')

    print(f"İşlem tamamlandı!")
    print(f"Yeni dosya: {cikis_dosya}")
    print(f"Toplam {len(satirlar)} satır işlendi.")
    print(f"{degisiklik_sayisi} satırda değişiklik yapıldı.")

except FileNotFoundError:
    print(f"HATA: '{dosya_yolu}' dosyası bulunamadı!")
    print("Dosya adını kontrol edin ve bu Python dosyasıyla aynı klasörde olduğundan emin olun.")
except Exception as e:
    print(f"Hata oluştu: {e}")