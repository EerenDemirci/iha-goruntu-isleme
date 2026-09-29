# Kaggle eğitim kodları

Eğitimler Kaggle'ın ücretsiz **Tesla T4** GPU'sunda yapıldı. Veri seti zaten Kaggle'da barındığı için 13 GB'lık veri hiç indirilmedi; yalnızca eğitilmiş model dosyaları (5,5 MB) indirildi.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `egitim_pipeline.py` | Temizlenmiş, yeniden çalıştırılabilir eğitim betiği (v1 / v3 / v4) |
| `kaggle_notebook_ham.ipynb` | Kaggle'da çalıştırılan notebook'un ham kaydı (deneme hücreleriyle birlikte) |

## Kullanım

1. Kaggle'da yeni bir notebook aç, veri setini bağla:
   [Aerial UAV Thermal – Inferred Unified Dataset](https://www.kaggle.com/datasets/umuttuygurr/aerial-uav-thermal-inferred-unified-dataset)
2. **Settings → Accelerator → GPU** ve **Settings → Internet → On** (ikisi de telefon doğrulaması ister)
3. `egitim_pipeline.py` dosyasını notebook'a yükle (sağ panel → Upload) ve çalıştır:

```python
!pip install -q ultralytics

SURUM = "v4"                                   # v1 | v3 | v4
exec(open("egitim_pipeline.py").read())
```

4. Eğitim bitince `/kaggle/working/egitim/deneme_v4/weights/best.pt` dosyasını indir.

## Deney tasarımı

Üç sürüm, **tek değişkenli** bir zincir oluşturur:

| Sürüm | Gerçek foto | İHA yapıştırması | Karşı örnek | Amaç |
|---|---|---|---|---|
| v1 | 3.000 | 0 | 0 | Temel model |
| v3 | 3.000 | 1.500 | 0 | Kestirme öğrenmeyi kırmak |
| v4 | 3.000 | 1.500 | 750 | Yan etkiyi gidermek |

Diğer her şey sabit: `yolo11n`, 20 epoch, batch 32, imgsz 640 ve `random.seed(42)` ile **aynı 3.000 gerçek fotoğraf**. Böylece sonuçlardaki fark yalnızca yapay veriden kaynaklanır.

## Betiğin adımları

1. **Gerçek veri** — kısayollarla (`os.symlink`) hazırlanır; 3.000 fotoğraf kopyalansa ~2 GB gereksiz yer kaplardı.
2. **Kesit toplama** — hangi fotoğrafta hangi sınıf olduğu `manifest.csv`'den okunur, yalnızca ilgili fotoğraflar açılır. (Önce 20.000 dosyayı tek tek taramıştık, dakikalar sürüyordu.)
3. **Yapay veri üretimi** — arka plan, konum, boyut ve kesit rastgele seçilir; etiket, yapıştırma koordinatından otomatik yazılır (piksel → oran). Arka planın kendi etiketleri korunur.
4. **`data.yaml`** — yollar ve sınıf isimleri. Etiket klasörü yazılmaz: YOLO yoldaki `images` kelimesini `labels` ile değiştirip kendisi bulur.
5. **Eğitim** — `yolo11n.pt` hazır ağırlıklarından devam (fine-tuning). Mimari değişmez; yalnızca çıkış katmanı 80 → 7 sınıfa iner (`Transferred 451/499 items from pretrained weights`).

## Notlar

- Kaggle notebook'larında **internet varsayılan olarak kapalıdır**; `pip install` için açmak gerekir.
- Oturum kapandığında `/kaggle/working` silinir. Eğitilmiş modeli bitince hemen indirmek gerekir.
- Bağlı veri setinin klasörü **salt okunurdur**; bu yüzden eğitim verisi `/kaggle/working` altında kurulur.
