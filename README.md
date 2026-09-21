# İHA Görüntü İşleme — OpenCV Öğrenme Projesi

Staj kapsamında OpenCV ile görüntü işlemeyi sıfırdan öğrenip **YOLO ile İHA görüntülerinde nesne tespiti** yapmaya ilerlediğim çalışma.

**Başlangıç:** 15 Eylül 2026 · **Repo açılışı:** 21 Eylül 2026

**Takip için:**
- 🗺️ **Yol haritası** → aşağıda
- 📓 **Öğrenme günlüğü** (her gün ne öğrendim, nerede zorlandım) → [`GUNLUK.md`](GUNLUK.md)
- 🧪 **Kendi yazdığım kodlar** → [`notebooks/00_calisma.ipynb`](notebooks/00_calisma.ipynb)
- 📦 **Veri seti** → [`data/README.md`](data/README.md)

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

## Veri

[Aerial UAV Thermal – Inferred Unified Dataset](https://www.kaggle.com/datasets/umuttuygurr/aerial-uav-thermal-inferred-unified-dataset) (CC0). 27.925 kuşbakışı, İHA ve termal görüntü; 7 sınıf için YOLO etiketleri var.
Tamamı 13 GB. Yerelde yalnızca küçük bir alt küme (`data/`, git'e girmez) kullanılır, tam eğitim Kaggle Notebook'ta (GPU) yapılır.

## Yol haritası (hedef: YOLO ile nesne tespiti)

| # | Aşama | Konu |
|---|-------|------|
| 1 | Görüntü temelleri ✅ | piksel, shape, dilimleme, renkli görüntü/BGR, dosyadan okuma, kutu çizme |
| 2 | YOLO etiket formatı ✅ | `sınıf x y g y` satırını piksel koordinatına çevirme, etiketleri resmin üstüne çizme |
| 3 | Veri setini tanıma | sınıf dağılımı, kutu boyutları, RGB ve termal farkı, train/val/test |
| 4 | Nesne tespiti kavramları | sınırlayıcı kutu, IoU, güven skoru, NMS, precision/recall, mAP |
| 5 | Hazır YOLO modeli | `ultralytics` ile tahmin, sonuçları okuma ve çizme |
| 6 | Eğitim (Kaggle GPU) | `data.yaml`, epoch, loss eğrileri, overfitting |
| 7 | Değerlendirme | hata analizi, küçük nesneler, termal görüntülerdeki performans |

## Klasör yapısı

```
notebooks/
  00_calisma.ipynb             # adım adım kendi yazdığım denemeler
  01_goruntu_numpy_dizisi.ipynb  # hazır ders notu: görüntü = NumPy dizisi
data/                          # veri seti (git'e girmez, bkz. data/README.md)
outputs/                       # üretilen görüntüler (git'e girmez)
GUNLUK.md                      # öğrenme günlüğü
```

Not: `00_calisma.ipynb`'deki bazı hücreler `outputs/ders01_sahne.png` ve `outputs/ders01_sahne.txt` (örnek YOLO etiketi) dosyalarını okur. `outputs/` git'e girmediği için bu dosyalar repoda yok. Görüntü `01_goruntu_numpy_dizisi.ipynb` çalıştırılınca üretilir.
