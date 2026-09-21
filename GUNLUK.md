# Öğrenme Günlüğü

Her çalışma gününde ne öğrendiğimi, nerede zorlandığımı ve sırada ne olduğunu buraya yazıyorum.

---

## 21 Eylül 2026 — Aşama 1: Görüntü temelleri ✅

**Kurulum**
- Python sanal ortamı (`.venv`), OpenCV 5.0, NumPy, Matplotlib, Jupyter
- VS Code'da notebook açma ve kernel seçme

**Öğrendiklerim**
- Görüntü bir sayı tablosudur, her sayı bir pikseldir. 0 siyah, 255 beyaz, arası gri.
- `shape`: gri görüntüde `(yükseklik, genişlik)`, renkli görüntüde `(yükseklik, genişlik, 3)`
- Tek piksel okuma ve değiştirme: `resim[satır, sütun]`. Sayma 0'dan başlar, son indeks boyuttan 1 eksiktir.
- Tablonun dışına çıkınca `IndexError` alınır. Hata mesajını okumaya son satırdan başlanır.
- Dilimleme ile bölge seçme: `resim[a:b, c:d]`. Başlangıç dahildir, bitiş dahil değildir (adet = bitiş − başlangıç).
- Matplotlib, `vmin`/`vmax` verilmezse renkleri en küçük ve en büyük değere göre otomatik ölçekler. Gri görüntülerde `vmin=0, vmax=255` kullanılır.
- Renkli görüntüde her pikselin 3 kanalı vardır. **OpenCV B-G-R sırası kullanır**, Matplotlib R-G-B. Göstermeden önce `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` ile çevrilir.
- `cv2.imread` ile dosyadan görüntü okuma
- `cv2.rectangle` ile kutu çizme. Çizim fonksiyonlarında sıra **(x, y)**, yani indekslemenin tersi. Çizim görüntüyü yerinde değiştirir, orijinali korumak için `.copy()` kullanılır.

**Zorlandığım yerler**
- İlk denediğim uzun, çok konulu notebook fazla yoğundu. Tek kavram, kısa kod ve önce tahmin etme şeklinde ilerlemeye geçtik, bu yöntem daha iyi oturdu.
- Dilimlemede "bitiş dahil değil" kuralı ve piksel numarasını dilime çevirmek başta karıştı. Alıştırmalarla oturdu.
- İndekslemede `[satır, sütun]`, çizimde `(x, y)` sırası kullanılması dikkat istiyor.

**Veri seti kararı**
- İki Kaggle veri setini karşılaştırdık. Hedef YOLO olduğu için 7 sınıflı, YOLO etiketli *Aerial UAV Thermal – Inferred Unified Dataset* seçildi.
- Veri seti 13 GB olduğu için tamamını indirmeyeceğim. Yerelde küçük bir alt kümeyle öğreneceğim, eğitimi Kaggle GPU'sunda yapacağım.

**Sıradaki:** Aşama 2, YOLO etiket formatı (`sınıf x_merkez y_merkez genişlik yükseklik` satırını piksel koordinatına çevirip kutuyu çizmek)
