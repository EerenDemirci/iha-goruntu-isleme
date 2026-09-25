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

### Aynı gün devam — Aşama 3: Gerçek veriyi tanıma (devam ediyor)

**Kaggle'da yaptıklarım**
- Kaggle hesabı açtım. Veri setine bağlı bir notebook açtım, 13 GB'lık verinin hiçbir kısmını indirmedim.
- Klasör yapısı: `images/` ve `labels/`, ikisinin içinde de `train/val/test`. Fotoğraf ile etiket aynı adı taşıyor, uzantıları `.png` ve `.txt`.
- `data.yaml` sınıfları tanımlıyor: `0 person, 1 car, 2 bicycle, 3 other_vehicle, 4 drone, 5 mine, 6 gun`
- `manifest.csv` ile verinin özetini çıkardım. **Dengesizlik** var: fotoğrafların %55'i tek bir kaynaktan (LLVIP) geliyor, nesnelerin %68'i `person`, `other_vehicle` ise yalnızca 148 tane. Modelin az örneği olan sınıflarda zayıf kalması bekleniyor.
- train / val / test ayrımının amacı: model train ile öğreniyor, val ile eğitim sırasında kontrol ediliyor, test ile hiç görmediği verilerde sınanıyor.
- Her kaynaktan ve her sınıftan örnek içeren **402 fotoğraflık (107 MB) bir alt küme** hazırlayıp indirdim. 7 sınıfın hepsi var.

**Kendi bilgisayarımda yaptıklarım** (`notebooks/01_gercek_veri.ipynb`)
- İlk gerçek görüntü: termal kamerayla kuşbakışı çekilmiş bir otopark, 640 × 512 piksel, 21 nesne (20 car, 1 other_vehicle)
- Fotoğraf boyutunu elle yazmak yerine `foto.shape` ile okudum. Boyut yanlış verilirse kutular kayıyor.
- Dosya yolunu metin birleştirerek (`+`) kurdum. Tek bir `ad` değişkeniyle hem fotoğrafa hem etikete ulaşılıyor.
- Etiketleri `yolo_kutu` fonksiyonumla gerçek fotoğrafa çizdim, kutular arabaları doğru sarıyor.
- **Önemli fark:** Aynı etiketleri boş, siyah bir resme çizince kutular yine aynı yerlerde çıktı. Yani kodum arabaları *bulmuyor*, insanların önceden çizdiği etiketleri *okuyor*. YOLO'nun öğreneceği şey tam olarak bu: etiket olmadan, sadece fotoğrafa bakarak kutuları kendisi bulmak.
- Sözlük (`dict`) ile sınıf numarasından sınıf adına ve renge ulaştım, `int()` ile metni tam sayıya çevirdim.

**Zorlandığım yerler**
- Yeni notebook önceki notebook'taki fonksiyonları bilmiyor. Hücreler yukarıdan aşağı çalıştırılmalı (`NameError: name 'cv2' is not defined`).
- Çok adım bir arada verilince takip etmek zorlaştı. Tek adım tek adım ilerlemek daha iyi.

**Sıradaki:** Kutulara sınıf adını yazmak (`cv2.putText`), farklı kaynaklardan (İHA, silah, mayın) fotoğraflara bakmak, sınıf başına kutu boyutlarını incelemek.

---

## 22 Eylül 2026 — Aşama 3 devam: kaynakları inceleme ve ilk veri analizi

**Tekrar**
- Güne dünkü konuların tekrarıyla başladım: `[y, x]` ile `(x, y)` farkı, BGR, `int`/`float`, `split` ve liste indeksleri, fonksiyon, döngü, dosya okuma, sözlük. Unuttuğum yerleri tekrarla oturttum.

**Yaptıklarım** (`notebooks/02_etiket_gorsellestirme.ipynb`)
- Kutuların üstüne sınıf adı yazmak için `cv2.putText` kullandım. Yazıyı kutunun üstüne koymak için `y1 - 3` gerekiyor, çünkü resimde y yukarıdan aşağı büyüyor.
- Fotoğrafı okuyup etiketlerini çizen işi tek bir fonksiyonda topladım: `etiketleri_ciz(ad, bolum)`. Artık her fotoğraf tek satırla görselleştirilebiliyor.
- Karışan eski notebook yerine temiz bir sayfa açtım. Ders: bir hücreyi düzeltince tekrar çalıştırmak gerekiyor, notebook en son çalıştırılan hali hatırlıyor.
- 6 kaynağın her birinden bir fotoğrafa baktım.
- `os.listdir`, biriktirme (`toplam = toplam + ...`), `append` ve iç içe döngü ile 260 train etiket dosyasındaki 1.062 nesneyi okudum. Her sınıfın tipik kutu boyutunu (ortanca kenar, piksel) hesaplayıp grafiğe döktüm.

**Bulgular**

| # | Bulgu | Model için anlamı |
|---|---|---|
| 1 | Sınıf dengesizliği: nesnelerin %68'i person, other_vehicle sadece 148 | Az örnekli sınıflarda zayıf kalabilir |
| 2 | Her kaynak belirli sınıflarda uzmanlaşmış (mayın → munitions, İHA → uav2uav, bisiklet → hituav) | Model arka plana bakıp "kestirme" öğrenebilir |
| 3 | Bütün görüntüler termal, renk bilgisi yok | Renkli fotoğraflarla eğitilmiş hazır modeller zayıf kalabilir. Kendi eğitimimiz gerekecek. |
| 4 | Tipik kutu kenarı: bicycle 25 px, mine 35 px, drone 44 px, gun 53 px, car 56 px, other_vehicle 59 px, person 92 px | En küçük nesneler bicycle ve mine |
| 5 | LLVIP'te otomatik çizilmiş bir araba kutusu iki arabanın arasında kalmış gibi | Otomatik etiketler hatalı olabilir ("çöp girerse çöp çıkar") |

- **Önemli ders:** Tek bir uav2uav fotoğrafına bakıp "en küçük nesneler İHA'lar" diye tahmin etmiştim. Sayılar en küçüğün bicycle olduğunu gösterdi. Tek örnekten genelleme yapmamak, gözlemi sayıyla doğrulamak gerekiyor.
- **Tahminim:** Eğitilen model en çok bicycle, mine ve other_vehicle sınıflarında zorlanacak. Aşama 7'de kontrol edeceğim.

**Zorlandığım yerler**
- İç içe döngü. Okul yoklaması benzetmesiyle (dış döngü = her sınıf, iç döngü = sınıftaki her öğrenci) daha iyi anladım.
- Çok adım bir arada verilince yoruldum. Küçük adımlarla ve beklenen çıktıyı önceden görerek ilerlemek daha iyi oldu.

**Sıradaki:** Aşama 3'ü kapatmak (val/test bölümlerine de bakmak, istersem boyutları orana göre karşılaştırmak), sonra Aşama 4: nesne tespiti kavramları (IoU, güven skoru, precision/recall).

---

## 23 Eylül 2026 — Aşama 4 (IoU ve başarı ölçme) ve Aşama 5 (hazır model)

**Tekrar ve alıştırma**
- Güne dünkü kodun üzerinden tekrarla başladım: sözlük ile liste farkı, `replace` (ekleme değil silme), `cv2.imread` ile `open`'ın neden ayrı olduğu, `int(satir.split()[0])` satırının içeriden dışarıya okunuşu, `( )` fonksiyon çağırır / `[ ]` eleman seçer.
- Sıfırdan yazma alıştırması (listedeki sayıların toplamı). Algoritmayı sözle eksiksiz kurdum, sözdiziminde takıldım. Tespit: mantık tamam, Python yazımı pratik istiyor.

**Aşama 4: Başarı nasıl ölçülür?** (`notebooks/03_iou_ve_basari.ipynb`)
- **IoU = kesişim / birleşim.** Elle hesapladım (5 px kayma → 0.33, 2 px kayma → 0.67), sonra `iou(a, b)` fonksiyonunu yazdım.
- Kesişimin kenarları `max`/`min` ile bulunur: ortak alan **geç başlayanda** başlar, **erken bitende** biter. `max(0, ...)` örtüşme yoksa alanı sıfıra çeker.
- Birleşimde ortak alan bir kez çıkarılır, yoksa iki kez sayılır.
- Gerçek veride kullandım: LLVIP fotoğrafındaki 3 araba kutusundan ikisi **%27** örtüşüyor. Dünkü gözlemimi ölçtüm.
- **TP / FP / FN**, **precision** (alarm verdiğinde ne kadar haklı) ve **recall** (var olanların kaçını yakaladı).
- **Güven skoru ve eşik**: eşik düşerse recall ↑ precision ↓, yükselirse tersi. İHA savunması gibi kaçırmanın pahalı olduğu yerde eşik düşük tutulur.
- **mAP**: bütün eşiklerin ortalaması, modelin karne notu (`mAP50`, `mAP50-95`).

**Aşama 5: Hazır YOLO modeli** (`notebooks/04_hazir_model.ipynb`)
- `ultralytics` kuruldu, hazır `yolo11n` modeli (COCO ile eğitilmiş, 80 sınıf) yüklendi.
- **Deneme 1 — kuşbakışı termal otopark** (gerçekte 21 nesne): model 4 kutu buldu, hepsi yanlış → *cell phone*, *bottle*, güven 0.27–0.35.
- **Deneme 2 — sokak seviyesinden termal görüntü** (gerçekte 3 araba + 5 insan): 5 kutu, *car* (0.93 / 0.91 / 0.77) ve *person* (0.42 / 0.36). Büyük ölçüde doğru.
- **Sonuç — alan farkı (domain gap):** COCO'da arabalar hep yandan görünür. Tepeden bakınca araba sadece parlak bir dikdörtgen olduğu için model en yakın bildiği şeye, cep telefonuna benzetiyor. Ayrıca kuşbakışı fotoğraflarda nesneler çok küçük.
- Hazır model ayrıca drone / mine / gun sınıflarını hiç tanımıyor. **Kendi modelimizi eğitmemizin gerekçesi bu.**

**Düzen**
- Notebook'lara adım başlıkları ve açıklama hücreleri eklendi, hangi hücrede ne yapıldığı yazılı.
- Değişken adları bütün sayfalarda ortaklaştırıldı: `ad`, `bolum`, `foto`, `yukseklik`, `genislik`, `etiketler`, `satir`, `numara`, `x1..y2`, `cizim`, `model`, `sonuc`, `kutu`.

**Sıradaki:** Aşama 6 — Kaggle GPU'sunda kendi modelimizi eğitmek.

---

## 24 Eylül 2026 — Aşama 6: Kendi modelimi eğittim 🎉

**Isınma:** precision/recall tekrarı. Payları aynı (doğru tespit), fark paydada: precision modelin *çizdiklerine*, recall *gerçekte var olanlara* bakar.

**Kaggle kurulumu**
- GPU'lu notebook açtım (Tesla T4). GPU ve internet için telefon doğrulaması gerekiyor.
- `pip install ultralytics` önce çalışmadı: Kaggle notebook'larında **internet varsayılan olarak kapalı**. Settings → Internet açınca düzeldi.
- Veri setini kısayolla (symlink) hazırladım: 3.000 train + 600 val fotoğraf. Kopyalama yok, 5 GB yerine sıfır ek alan.
- `data.yaml` yazdım (yollar + 7 sınıf). YOLO etiketleri `images` yerine `labels` klasöründe arar.

**Eğitim**
- `yolo11n.pt` hazır modelinden başladım (**transfer öğrenme**), 20 epoch, imgsz 640, batch 32.
- Süre: **20 dakika**. 3000 ÷ 32 = 94 adım/epoch × 20 = 1.880 düzeltme adımı.
- Eğitim döngüsü: tahmin → gerçek etiketle karşılaştır → loss → 2,5 milyon parametreyi azıcık düzelt.

**Sonuç: mAP50 = 0.789, mAP50-95 = 0.554** (ilk epoch'ta mAP50 0.37'ydi)

| Sınıf | mAP50 | Tahminim tuttu mu? |
|---|---|---|
| mine | 0.991 | ❌ "zor olur" demiştim, en iyi çıktı |
| gun | 0.978 | |
| drone | 0.938 | |
| person | 0.930 | |
| car | 0.926 | |
| bicycle | 0.627 | ✅ "en küçük nesneler, zor" |
| other_vehicle | 0.133 | ✅ "148 örnekle öğrenilmez" — recall 0 |

**Grafiklerden okuduklarım** (`egitim_sonuclari/`)
- `results.png`: train ve val loss birlikte düşüyor → **ezberleme yok**. mAP eğrileri 20. epoch'ta hâlâ yükseliyor → **daha uzun eğitim daha iyi sonuç verir**.
- `confusion_matrix.png`: `other_vehicle` için model hiç tahmin yapmamış, gerçekteki 6 nesnenin 4'üne **car** demiş. Sınıflar arası başka karışıklık yok; hatalar çoğunlukla yanlış alarm (248 person) ve kaçırma (114 person).
- Precision eğrisi ilk epoch'ta 0.98'den 0.74'e düşüyor: model başta ürkekti, az kutu çizip hepsini tutturuyordu; sonra çok kutu çizmeye başladı, recall 0.14'ten 0.72'ye çıktı.

**Hazır modelle karşılaştırma** (`notebooks/05_kendi_modelim.ipynb`)

| Fotoğraf | Hazır model | Benim modelim |
|---|---|---|
| Kuşbakışı otopark (21 nesne) | 4 kutu: *cell phone*, *bottle* | **23 araç**, çoğu 0.90+ güven |
| Sokak termal (8 nesne) | 3 car + 2 person | **4 car + 5 person** |
| Gökyüzünde İHA (2 nesne) | sınıfı bilmiyor | **2 drone** |

**Sıradaki adımlar**
1. `other_vehicle` için daha fazla örnek, ya da `car` ile birleştirmek
2. Daha uzun eğitim (50+ epoch) ve daha çok veri
3. Kestirme öğrenme testi: her sınıf tek bir kaynaktan geldiği için model nesneyi mi, fotoğrafın türünü mü tanıyor?

---

## 25 Eylül 2026 — Aşama 7, ikinci eğitim ve video

**Tekrar:** precision/recall alıştırması (10 kutu, 7 doğru, 14 gerçek nesne → P 0.7, R 0.5). Payları aynı, fark paydada: precision modelin *çizdiklerine*, recall *gerçekte var olanlara* bakar.

### Aşama 7 — Değerlendirme (`notebooks/06_degerlendirme.ipynb`)

**1) Test bölümünde ölçüm.** Val eğitim sırasında her epoch kullanıldı ve `best.pt` ona göre seçildi; `test` bölümüne hiç dokunulmadı.

| | val | test |
|---|---|---|
| mAP50 | 0.789 | **0.800** |

Test val'den kötü değil → **ezberleme yok**, model genelleşiyor. (Bu ölçüm yerel 72 fotoğrafla yapıldı.)

**2) Kestirme öğrenme testi.** Aynı İHA parçasını farklı arka planlara yapıştırdım. Nesne birebir aynı, değişen tek şey arka plan:

| Deney | Sonuç |
|---|---|
| Kendi gökyüzüne yapıştır (kontrol) | drone 0.89 ✅ |
| Otopark asfaltına yapıştır | hiçbir şey ❌ |
| Asfalt, güven eşiği 0.01 | `car 0.03` — İHA sezgisi yok, zayıf *araba* sezgisi var |
| Elle üretilmiş düz koyu tuval | **drone 0.91** ✅ |
| Asfalt, İHA 2 kat büyütülmüş | hiçbir şey ❌ (sorun boyut değil) |

**Modelin öğrendiği kural:** *"Düz ve koyu bir arka planda küçük parlak bir leke varsa, bu bir İHA'dır."* Nesnenin şekli bu kuralın küçük bir parçası, arka plan baskın. Sebep veri: bütün İHA'lar `uav2uav` kaynağından, hepsi gökyüzü arka planlı.

**3) Klasik yöntem karşılaştırması.** Sadece OpenCV: gri tonlama → Otsu eşikleme → morfolojik açma → kontur → boyut süzgeci.

| | Klasik | YOLO |
|---|---|---|
| İHA fotoğrafı (2 nesne) | 2/2 | 2/2 |
| **Otopark (21 nesne)** | **3/21** | **21/21** |
| Sınıf ayrımı | yok, sadece "leke" | 7 sınıf |
| Hız | 4 ms | ~7-12 ms |
| Eğitim | gerekmez | 20 dk GPU + etiketli veri |

Otoparkta varsayım ters döndü: termal görüntüde güneşte ısınan asfalt, arabalardan daha parlak. "Parlak olan nesnedir" kuralı çöktü.

### İkinci eğitim (Kaggle, Tesla T4)

10.000 train + 2.000 val, 40 epoch, `patience=15`, 2,6 saat.

| | 1. eğitim | **2. eğitim** |
|---|---|---|
| Veri | 3.000 / 600 | 10.000 / 2.000 |
| Epoch | 20 | 40 |
| mAP50 | 0.789 | **0.857** |
| mAP50-95 | 0.554 | **0.642** |
| Recall | 0.719 | **0.816** |

| Sınıf | 1. eğitim | 2. eğitim |
|---|---|---|
| mine | 0.991 | 0.995 |
| drone | 0.938 | **0.980** |
| gun | 0.978 | 0.977 |
| person | 0.930 | **0.967** |
| car | 0.926 | **0.959** |
| bicycle | 0.627 | **0.798** |
| other_vehicle | 0.133 | **0.323** |

En çok iyileşenler tam da zorlandığımız sınıflar. `other_vehicle` hâlâ zayıf ama artık recall 0 değil. Teşhis doğruydu: sorun veri azlığıydı.

### Video (`notebooks/07_video.ipynb`)

- `cv2.VideoWriter` ile kendi test videomu ürettim: gökyüzünde hareket eden bir İHA, 100 kare / 20 FPS.
- `cv2.VideoCapture` ile kare kare okuyup her kareye model uyguladım, kutulu halini ikinci videoya yazdım.
- **Gökyüzü arka planı: 100/100 karede drone bulundu. Otopark arka planı: 0/100.** Aynı İHA, aynı hareket, tek fark arka plan.
- Parlak zemin denemesi: nesne ile arka plan aynı parlaklıkta olduğunda klasik yöntem **kuralı basit olduğu için**, YOLO **yanlış şeyi öğrendiği için** başarısız oluyor.

**YOLO nasıl çalışıyor (öğrendiklerim):** omurga (kenar → doku → şekil), boyun (farklı ölçekleri birleştirir), baş (ızgara hücreleri için kutu + 7 sınıf puanı). Güven eşiği ve NMS ile süzülür. Son katmanlarda bir hücrenin **alıcı alanı** neredeyse tüm görüntüdür — bağlamın bu kadar etkili olmasının sebebi bu.

**Sıradaki:** Yeni modeli (`deneme2/best.pt`) indirip karşılaştırmak, veri setindeki ardışık karelerden gerçek termal video üretmek, nesne takibi (tracking).

**Gün sonu ek — v2 modeli indirildi ve karşılaştırıldı** (`notebooks/08_gelistirilmis_model.ipynb`)

| Fotoğraf (gerçek) | v1 | v2 |
|---|---|---|
| Otopark (20 car + 1 other_vehicle) | 23 kutu, hepsi `car` | 22 kutu: **21 car + 1 other_vehicle** ✅ |
| Sokak (3 car + 5 person) | 4 car + 5 person | 4 car + 5 person |
| Gökyüzü (2 drone) | 2 drone | 2 drone |

v2, v1'in kamyonete "car" dediği yerde doğru sınıfı buluyor.

**Kestirme öğrenme testi tekrarlandı: düzelmemiş.** Her iki model de İHA'yı düz koyu tuvalde tanıyor (0.91 / 0.88), asfaltta hiç tanımıyor. Eklenen 7.000 fotoğrafta da bütün İHA'lar gökyüzündeydi; "gökyüzü = İHA" kuralı daha da pekişti.

**Ders:** Veri miktarı her sorunu çözmüyor. Az örnekli sınıflar miktardan fayda gördü, kestirme öğrenme ise bir **çeşitlilik** sorunu olduğu için görmedi.

v2 eğrileri v1'den farklı olarak sona doğru yatıklaşmış: aynı kurulumla daha uzun eğitmek büyük kazanç getirmez, bundan sonrası veri çeşitliliği ya da daha büyük modelle gelir.
