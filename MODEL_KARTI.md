# Model Kartı — Fine-tuned YOLO11n (Termal İHA Tespiti)

*Model kartı, bir modelin ne yaptığını, hangi veriyle eğitildiğini, nerede çalıştığını ve **nerede çalışmadığını** tek sayfada anlatan belgedir. Amaç, modeli kullanacak kişinin sınırları bilmeden karar vermesini önlemektir.*

---

## 1. Genel bilgi

| | |
|---|---|
| **Model** | YOLO11n (nesne tespiti) |
| **Yöntem** | Fine-tuning — COCO ağırlıklarının üzerine ince ayar. **Mimari değiştirilmedi**; tek yapısal değişiklik çıkış katmanının 80 → 7 sınıfa inmesi |
| **Parametre sayısı** | 2,58 milyon |
| **Model dosyası** | 5,5 MB |
| **Sınıflar (7)** | `person` · `car` · `bicycle` · `other_vehicle` · `drone` · `mine` · `gun` |
| **Girdi** | Termal / kızılötesi görüntü, 640×640'a ölçeklenir |
| **Çıkarım hızı** | 3,0 ms/görüntü (Tesla T4) → ~330 FPS · 12 ms (Apple M2, MPS) |
| **Eğitim süresi** | 20 epoch, ~35 dakika (Tesla T4) |
| **Geliştirme dönemi** | 15–29 Eylül 2026 (staj çalışması) |

## 2. Kullanım amacı

**Amaçlanan kullanım:** Termal/kızılötesi görüntülerde nesne tespiti üzerine bir **öğrenme ve araştırma çalışması**. Özellikle kestirme öğrenmenin (shortcut learning) tespiti ve yapay veriyle giderilmesi üzerine bir vaka çalışması.

**Amaçlanmayan kullanım:** Bu model **gerçek bir güvenlik, savunma veya karar destek sisteminde kullanılmamalıdır**. Tek bir veri setiyle, sınırlı çeşitlilikte ve kısa sürede eğitilmiştir; aşağıdaki sınırlar bu kararı zorunlu kılar.

## 3. Eğitim verisi

| | |
|---|---|
| **Kaynak** | [Aerial UAV Thermal – Inferred Unified Dataset](https://www.kaggle.com/datasets/umuttuygurr/aerial-uav-thermal-inferred-unified-dataset) (CC0) |
| **Toplam** | 27.925 görüntü, 6 alt kaynaktan birleştirilmiş |
| **Kullanılan** | 3.000 gerçek (eğitim) + 600 doğrulama + 2.250 yapay üretilmiş |
| **Etiketler** | YOLO formatı. Bir kısmı (LLVIP araç kutuları) **otomatik üretilmiş**, hatalı olabilir |

**Bilinen veri sorunları:**
- **Sınıf dengesizliği:** nesnelerin %68'i `person`, `other_vehicle` yalnızca 148 örnek
- **Kaynak–sınıf bağı:** her sınıf ağırlıklı olarak tek bir alt kaynaktan geliyor (İHA'lar hep gökyüzü arka planlı, mayınlar hep benzer toprak zeminde)
- **Etiket gürültüsü:** LLVIP araç kutularının bir kısmı bir modelle üretilmiş; gözle incelemede iki aracın arasına düşmüş kutular görüldü

## 4. Başarı ölçümleri

**Tam test kümesi** (2.798 görüntü, 8.748 nesne — eğitimde ve model seçiminde hiç kullanılmadı):

| Sınıf | Nesne | mAP50 | Recall |
|---|---|---|---|
| mine | 182 | 0.992 | 0.989 |
| gun | 187 | 0.952 | 0.963 |
| person | 5.930 | 0.943 | 0.933 |
| car | 1.694 | 0.931 | 0.911 |
| drone | 367 | 0.931 | 0.935 |
| bicycle | 378 | 0.648 | 0.758 |
| other_vehicle | 10 | 0.435 | 0.300 |
| **Genel** | **8.748** | **0.833** | 0.827 |

**Ölçüm uyarısı:** `other_vehicle` için test kümesinde yalnızca 10 nesne var; bu sınıfın sayısı istatistiksel olarak güvenilir değildir. Aynı ölçüm 72 görüntülük küçük bir kümede yapıldığında bu sınıf **0.000** çıkıyordu — küçük örneklem hem iyimser hem kötümser yanılgı üretebilir.

## 5. Davranışsal testler (standart metriklerin göstermediği)

Model kutu doğruluğu iyi olsa da **nesneyi mi yoksa bağlamı mı tanıdığı** ayrı olarak sınandı. Yöntem: bir nesne kesilip farklı arka planlara yapıştırıldı, kararın değişip değişmediğine bakıldı.

| Test | v1 (yapay verisiz) | v3 (İHA yapıştırmalı) | **v4 (+ karşı örnek)** |
|---|---|---|---|
| İHA, gökyüzü olmayan 5 sahnede | 2/5 | 5/5 | **5/5** |
| İHA, otopark arka planında video (100 kare) | 1/100 | 100/100 | **100/100** |
| Görülmemiş İHA kesitleri (24 deneme) | 5/24 | 23/24 | **23/24** |
| Gökyüzüne konan araba doğru sınıflandı (24) | 11/24 | 2/24 | **22/24** |
| Gökyüzüne konan insan doğru sınıflandı (24) | 14/24 | 12/24 | **24/24** |

Ayrıntı: [`notebooks/09_yapay_veri.ipynb`](notebooks/09_yapay_veri.ipynb) ve [`notebooks/10_yan_etki.ipynb`](notebooks/10_yan_etki.ipynb)

## 6. Sınırlar

**Veri kaynağına bağımlılık.** Bütün görüntüler tek bir veri setinden, dolayısıyla sınırlı sayıda kamera ve çekim koşulundan geliyor. Farklı bir termal kamerada, farklı çözünürlük veya sıcaklık aralığında davranışı **test edilmedi**.

**Sadece termal.** Model normal renkli (RGB) görüntülerde denenmedi; eğitim verisinin tamamı termal/kızılötesi.

**Zayıf sınıflar.** `other_vehicle` (mAP50 0.435) ve `bicycle` (0.648) yetersiz. Sebep: az örnek ve küçük nesne boyutu. Bu iki sınıfın çıktısına güvenilmemeli.

**Kestirme öğrenme tamamen çözülmedi.** v4 ile iki yönlü olarak büyük ölçüde kırıldı, ancak:
- Karşı örnekler yalnızca 3 sınıf için üretildi (`person`, `car`, `bicycle`); `mine` ve `gun` için benzer test yapılmadı
- `mine` sınıfının mAP50'si 0.99 — bu kadar yüksek bir sonuç, bu sınıfın tek bir kaynaktan gelmesi nedeniyle bağlam ipucu içeriyor olabilir; ayrıca sınanmalı

**Yapay verinin kendi sınırları.** Yapıştırma kenarları keskin, ışık ve gölge uyumsuz, perspektif yok. Model "yama artefaktını" öğrenmiş olabilir; bu ayrıca ölçülmedi.

**Küçük nesneler.** Tipik kutu kenarı `bicycle` için 25 piksel, `mine` için 35 piksel. YOLO girdiyi 640 piksele ölçeklediği için bu nesneler daha da küçülür.

## 7. Etik ve güvenlik notları

- Model `gun` ve `mine` gibi **güvenlik açısından hassas sınıflar** içeriyor. Yanlış bir tespit (ya da kaçırma) gerçek bir sistemde ciddi sonuçlar doğurabilir. Bu model bu amaçla kullanılamaz.
- Eğitim verisi kamuya açık ve CC0 lisanslıdır. Çalışma boyunca **hiçbir kurumsal veya gizli görüntü kullanılmamıştır**.
- Güven eşiği uygulamaya göre ayarlanmalıdır: kaçırmanın maliyeti yüksekse eşik düşürülür (yanlış alarm artar), tersi durumda yükseltilir.

## 8. Yeniden üretilebilirlik

| | |
|---|---|
| **Eğitim kodu** | [`kaggle/egitim_pipeline.py`](kaggle/egitim_pipeline.py) |
| **Rastgelelik** | `random.seed(42)` (veri seçimi), `seed(7)` / `seed(8)` (yapay üretim) |
| **Ortam** | Kaggle notebook, Tesla T4, `ultralytics` 8.4.x, PyTorch 2.10 |
| **Deney tasarımı** | v1 → v3 → v4 zincirinde her adımda **tek değişken** değişir |

## 9. Sonraki adımlar

1. Farklı kaynaktan termal görüntülerle sınama (kamera genellemesi)
2. `other_vehicle` ve `bicycle` için veri artırımı ya da sınıf birleştirme
3. `mine` ve `gun` sınıfları için bağlam bağımlılığı testi
4. Yapıştırma artefaktının ölçülmesi (kenar yumuşatma, harmanlama ile karşılaştırma)
5. Adversarial examples ile dayanıklılık testi
