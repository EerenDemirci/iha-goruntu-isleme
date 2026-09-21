# Öğrenme Günlüğü

Her çalışma gününde ne öğrendiğimi, nerede zorlandığımı ve sırada ne olduğunu buraya yazıyorum.

---

## 15–20 Eylül 2026 — Başlangıç: OpenCV ve NumPy'a giriş

- Staj kapsamında görüntü işleme çalışmasına başladım.
- OpenCV'ye giriş ve NumPy temellerini kendi başıma çalıştım (bu repo açılmadan önce).
- Bu konuları pekiştirmem gerektiğini fark ettim. 21 Eylül'de repoyu açıp temelleri adım adım tekrar ederek ilerlemeye başladım.

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

### Aynı gün devam — Aşama 2: YOLO etiket formatı ✅

**Öğrendiklerim**
- YOLO etiket satırı: `sınıf merkez_x merkez_y genişlik yükseklik`. Değerler piksel değil, **0–1 arası oran**.
- Orandan piksele: yatay değerler resmin genişliğiyle, dikey değerler yüksekliğiyle çarpılır. Oran kullanılmasının sebebi resim boyutu değişse de etiketin geçerli kalması.
- Merkezden köşeye: `x1 = mx − g/2`, `x2 = mx + g/2` (y için de aynısı yükseklikle). `cv2.rectangle` köşe istediği için bu dönüşüm gerekiyor.
- `round()` ile tam sayıya yuvarlama. Ondalıklı sayılarda küçük bilgisayar hataları olur (`0.695 × 600 = 416.99999999999994`), bu da ±1 piksel farka yol açabilir. Normal bir durum.
- Metin ile sayı farkı: `"0.695" * 3` sonucu `"0.6950.6950.695"`, hata vermeden yanlış sonuç üretiyor. Çözüm `float()`.
- `.split()` ile satırı parçalara, `.splitlines()` ile metni satırlara ayırmak. Listelerde de indeks 0'dan başlar.
- İlk fonksiyonum: `yolo_kutu(satir, genislik, yukseklik)`. Etiket satırını `(x1, y1, x2, y2)` köşelerine çeviriyor (`def`, parametre, `return`).
- `for` döngüsü ile her etiketi sırayla çizmek
- `with open(...)` ile etiket dosyasını okumak. `\n` yeni satır karakteri, `type()` ve `repr()` ile inceleme.
- **Sonuç:** Resmi ve aynı isimli `.txt` etiket dosyasını okuyup bütün nesnelerin kutusunu çizen kodu yazdım.

**Zorlandığım yerler**
- Kodu sıfırdan yazmak. Kalıp verilince ve boşluk doldurunca daha rahat ilerledim.
- Döngünün içinde değişken yerine sabit metin kullandım, 4 tur da aynı kutuyu çizdi. Ders: döngü değişkeni kullanılmazsa "kod çalışıyor ama sonuç hep aynı" olur.
- Girinti: hangi satırın döngünün ya da `with` bloğunun içinde olduğunu belirlemek dikkat istiyor. Standart 4 boşluk.
- Etiket satırının nereden geldiğini sordum: örnekteki satırlar sahneden tersine hesaplandı, gerçek veri setinde insanlar ya da bir model tarafından üretilmiş hazır `.txt` dosyaları var.

**Sıradaki:** Aşama 3. Kaggle'da veri setinden küçük bir alt küme hazırlamak, gerçek bir fotoğraf ve etiket dosyasını açıp `yolo_kutu` ile çizmek, sınıf numaralarının anlamını veri setinin ayar dosyasından öğrenmek.
