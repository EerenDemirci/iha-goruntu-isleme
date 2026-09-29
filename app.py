"""
İHA / termal görüntü nesne tespiti — demo arayüzü

İki sekme var:
  1) Fotoğraf  — tek görüntüde nesne tespiti
  2) Video     — karelerde tespit + NESNE TAKİBİ (her nesneye bir kimlik verilir)

Çalıştırmak için:  python app.py
"""

import os
import tempfile
import time

import cv2
import gradio as gr
from ultralytics import YOLO


# =====================================================================
# 1) MODELLER
#    Birden çok model sunuyoruz ki fine-tuning'in etkisi arayüzde
#    doğrudan görülebilsin.
# =====================================================================
MODELLER = {
    "Fine-tuned YOLO11n (v4 — son sürüm)": "models/iha_yolo11n_v4.pt",
    "Fine-tuned YOLO11n (v1 — ilk sürüm)": "models/iha_yolo11n.pt",
    "Hazır YOLO11n (COCO, ince ayarsız)":  "models/yolo11n.pt",
}

_yuklu = {}                                    # fotoğraf sekmesi için yüklü modeller
_yuklu_video = {}                              # video sekmesi ayrı tutulur: takip durumu
                                               # (kimlikler) modelin içinde saklandığı için
                                               # iki sekmenin birbirini bozmasını istemiyoruz


def model_getir(ad, video=False):
    """Model dosyasını yükler ve bellekte tutar (ilk çağrıda yükler)."""
    kutu = _yuklu_video if video else _yuklu
    if ad not in kutu:
        kutu[ad] = YOLO(MODELLER[ad])
    return kutu[ad]


# =====================================================================
# 2) FOTOĞRAF SEKMESİ
# =====================================================================
def tespit_et(resim, model_adi, guven_esigi):
    """
    resim        : kullanıcının yüklediği görüntü (RGB, NumPy dizisi)
    model_adi    : açılır listeden seçilen model
    guven_esigi  : kaydırıcıdan gelen değer (0–1)

    Döndürür: kutulanmış görüntü, tespit tablosu, özet metni
    """
    if resim is None:
        return None, [], "Önce bir görüntü yükleyin."

    model = model_getir(model_adi)

    # Gradio RGB verir, OpenCV/YOLO BGR ile çalışır → çeviriyoruz
    bgr = cv2.cvtColor(resim, cv2.COLOR_RGB2BGR)

    sonuc = model(bgr, conf=guven_esigi, verbose=False)[0]      # tahmin

    # --- kutuları çiz ---
    cizim = sonuc.plot()                                        # BGR döner
    cizim = cv2.cvtColor(cizim, cv2.COLOR_BGR2RGB)              # Gradio için RGB'ye çevir

    # --- tespit tablosu ---
    satirlar = []
    sayim = {}
    for kutu in sonuc.boxes:
        sinif = model.names[int(kutu.cls)]
        guven = float(kutu.conf)
        x1, y1, x2, y2 = [round(v) for v in kutu.xyxy[0].tolist()]

        satirlar.append([sinif, f"{guven:.2f}", f"{x1}, {y1}", f"{x2 - x1} x {y2 - y1}"])
        sayim[sinif] = sayim.get(sinif, 0) + 1

    # güven skoruna göre büyükten küçüğe sırala
    satirlar.sort(key=lambda s: float(s[1]), reverse=True)

    # --- özet metni ---
    if sayim:
        dokum = ", ".join(f"{adet} {ad}" for ad, adet in sorted(sayim.items()))
        sure = sonuc.speed["inference"]
        ozet = (f"**{len(satirlar)} nesne bulundu:** {dokum}\n\n"
                f"Görüntü boyutu: {resim.shape[1]} x {resim.shape[0]} piksel · "
                f"İşlem süresi: {sure:.1f} ms · Güven eşiği: {guven_esigi}")
    else:
        ozet = (f"Hiçbir nesne bulunamadı. Güven eşiği **{guven_esigi}**; "
                f"düşürürseniz model daha çok kutu gösterir.")

    return cizim, satirlar, ozet


# =====================================================================
# 3) VİDEO SEKMESİ — TESPİT + TAKİP
#
#    Tespit  : her kareye ayrı ayrı bakar, "burada bir araba var" der.
#    Takip   : kareler arasında bağ kurar, "bu, önceki karedeki 3 numaralı
#              arabanın ta kendisi" der. Böylece nesne SAYABİLİRİZ:
#              100 karede 100 tespit değil, 1 nesne.
#
#    Kullandığımız takipçi ByteTrack: yeni karedeki kutularla önceki
#    karedeki kutuları IoU (örtüşme) benzerliğine göre eşleştirir.
# =====================================================================
MAKS_KARE = 900               # ~45 saniye (20 FPS) — demoda beklemeyi sınırlıyoruz


def video_isle(video_yolu, model_adi, guven_esigi, en_az_kare, ilerleme=gr.Progress()):
    """
    video_yolu   : yüklenen videonun dosya yolu
    en_az_kare   : bir kimlik en az kaç karede görülürse "gerçek nesne" sayılsın

    Döndürür: işlenmiş video, kimlik tablosu, özet metni
    """
    if not video_yolu:
        return None, [], "Önce bir video yükleyin."

    model = model_getir(model_adi, video=True)

    # Takipçi eski videodan kalma kimlikleri hatırlamasın diye sıfırlıyoruz
    if getattr(model, "predictor", None) is not None:
        for takipci in getattr(model.predictor, "trackers", []):
            takipci.reset()

    okuyucu = cv2.VideoCapture(video_yolu)
    if not okuyucu.isOpened():
        return None, [], "Video açılamadı."

    fps      = okuyucu.get(cv2.CAP_PROP_FPS) or 20.0
    genislik = int(okuyucu.get(cv2.CAP_PROP_FRAME_WIDTH))
    yukseklik = int(okuyucu.get(cv2.CAP_PROP_FRAME_HEIGHT))
    toplam   = int(okuyucu.get(cv2.CAP_PROP_FRAME_COUNT)) or MAKS_KARE

    # avc1 (H.264) tarayıcıda oynar; yoksa mp4v'ye düşüyoruz
    cikti_yolu = os.path.join(tempfile.mkdtemp(), "takip_sonucu.mp4")
    yazici = cv2.VideoWriter(cikti_yolu, cv2.VideoWriter_fourcc(*"avc1"),
                             fps, (genislik, yukseklik))
    if not yazici.isOpened():
        yazici = cv2.VideoWriter(cikti_yolu, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (genislik, yukseklik))

    # kimlik → [sınıf, kaç karede görüldü, en yüksek güven, ilk kare]
    kimlikler = {}
    kare_sayisi = 0
    baslangic = time.time()

    while kare_sayisi < MAKS_KARE:
        okundu, kare = okuyucu.read()
        if not okundu:
            break
        kare_sayisi += 1

        # persist=True: "önceki kareyi hatırla" → kimlikler kareler arasında korunur
        sonuc = model.track(kare, conf=guven_esigi, persist=True,
                            tracker="bytetrack.yaml", verbose=False)[0]

        if sonuc.boxes.id is not None:                   # takipçi kimlik atayabildiyse
            for kutu, kimlik in zip(sonuc.boxes, sonuc.boxes.id.int().tolist()):
                sinif = model.names[int(kutu.cls)]
                kayit = kimlikler.setdefault(kimlik, [sinif, 0, 0.0, kare_sayisi])
                kayit[1] += 1
                kayit[2] = max(kayit[2], float(kutu.conf))

        yazici.write(sonuc.plot())                       # kutu + kimlik çizilmiş kare
        ilerleme(kare_sayisi / min(toplam, MAKS_KARE), desc=f"Kare {kare_sayisi}")

    okuyucu.release()
    yazici.release()
    gecen = time.time() - baslangic

    # --- kısa ömürlü kimlikleri ele ---
    # Bir nesne yalnızca 1-2 karede görünüyorsa bu çoğunlukla yanlış alarm ya da
    # takipçinin kopan bir izidir; saymaya katmıyoruz.
    kalici = {k: v for k, v in kimlikler.items() if v[1] >= en_az_kare}

    satirlar = [[k, v[0], f"{v[2]:.2f}", v[1], v[3]] for k, v in kalici.items()]
    satirlar.sort(key=lambda s: -s[3])                   # en uzun görülen üstte

    sayim = {}
    for v in kalici.values():
        sayim[v[0]] = sayim.get(v[0], 0) + 1

    if sayim:
        dokum = ", ".join(f"{adet} {ad}" for ad, adet in sorted(sayim.items()))
        ozet = (
            f"**{len(kalici)} benzersiz nesne takip edildi:** {dokum}\n\n"
            f"{kare_sayisi} kare · {genislik} x {yukseklik} · "
            f"toplam {gecen:.1f} sn ({kare_sayisi / gecen:.1f} kare/sn)\n\n"
            f"*Sayım, tespit sayısı değil **kimlik** sayısıdır: aynı araba 100 karede "
            f"görünse de 1 nesne sayılır. Elenen kısa ömürlü iz: "
            f"{len(kimlikler) - len(kalici)}.*"
        )
    else:
        ozet = (f"{kare_sayisi} kare işlendi, kalıcı bir nesne bulunamadı. "
                f"Güven eşiğini ya da 'en az kare' değerini düşürmeyi deneyin.")

    return cikti_yolu, satirlar, ozet


# =====================================================================
# 4) ARAYÜZ
# =====================================================================
ACIKLAMA = """
# İHA ve termal görüntü nesne tespiti

Termal / kızılötesi görüntülerde **7 sınıf** tespit eder:
`person` · `car` · `bicycle` · `other_vehicle` · `drone` · `mine` · `gun`

Model: **YOLO11n**, [Aerial UAV Thermal veri setiyle](https://www.kaggle.com/datasets/umuttuygurr/aerial-uav-thermal-inferred-unified-dataset) fine-tune edilmiştir.
Mimari değiştirilmemiş; COCO ağırlıklarının üzerine ince ayar yapılmıştır (çıkış katmanı 80 → 7 sınıf).

**Model seçeneklerini karşılaştırın:** *Hazır YOLO11n* kuşbakışı termal görüntülerde arabalara "cell phone" der ve
`drone`, `mine`, `gun` sınıflarını hiç tanımaz. Fine-tuned sürümler bu görüntüler için eğitilmiştir.
"""

VIDEO_ACIKLAMA = """
### Videoda nesne takibi

Tespit her kareye tek tek bakar. **Takip**, kareler arasında bağ kurup her nesneye bir **kimlik (ID)** verir —
böylece nesneleri *sayabiliriz*: aynı araba 100 karede görünse bile 1 nesnedir.

Takipçi: **ByteTrack** — yeni karedeki kutuları öncekilerle örtüşmelerine (IoU) göre eşleştirir.
Bir nesne başka bir nesnenin arkasına girip çıkarsa kimliği değişebilir (*ID switch*); tablodaki kimlik sayısının
gerçek nesne sayısından biraz fazla çıkmasının sebebi budur.
"""

with gr.Blocks(title="İHA Tespit Demo") as arayuz:
    gr.Markdown(ACIKLAMA)

    with gr.Tabs():
        # ---------------- FOTOĞRAF ----------------
        with gr.Tab("Fotoğraf"):
            with gr.Row():
                with gr.Column():
                    girdi_resim = gr.Image(label="Görüntü yükleyin", type="numpy", height=380)
                    gr.Examples(
                        examples=[
                            "ornekler/otopark_kusbakisi.png",   # kuşbakışı termal otopark
                            "ornekler/gokyuzunde_iha.png",      # gökyüzünde 2 İHA
                            "ornekler/sokak_termal.png",        # sokak seviyesinden termal
                            "ornekler/mayin_sahasi.png",        # mayın sahası
                        ],
                        inputs=girdi_resim,
                        label="Örnek görüntüler (tıklayın)",
                        cache_examples=False,                   # tıklanınca canlı çalışsın
                    )
                    model_secim = gr.Dropdown(
                        choices=list(MODELLER.keys()),
                        value=list(MODELLER.keys())[0],
                        label="Model",
                    )
                    esik = gr.Slider(
                        minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                        label="Güven eşiği",
                        info="Düşürünce model daha çok kutu gösterir (kaçırma azalır, yanlış alarm artar).",
                    )
                    calistir = gr.Button("Tespit et", variant="primary")

                with gr.Column():
                    cikti_resim = gr.Image(label="Tespit sonucu", height=380)
                    ozet_metni = gr.Markdown()
                    tablo = gr.Dataframe(
                        headers=["Sınıf", "Güven", "Sol üst köşe (x, y)", "Boyut (piksel)"],
                        label="Bulunan nesneler",
                        wrap=True,
                    )

        # ---------------- VİDEO ----------------
        with gr.Tab("Video (nesne takibi)"):
            gr.Markdown(VIDEO_ACIKLAMA)
            with gr.Row():
                with gr.Column():
                    girdi_video = gr.Video(label="Video yükleyin", height=340)
                    gr.Examples(
                        examples=[
                            "ornekler/video_otopark.mp4",   # otoparkın üzerinde uçan İHA
                            "ornekler/video_gokyuzu.mp4",   # gökyüzünde İHA
                        ],
                        inputs=girdi_video,
                        label="Örnek videolar (tıklayın)",
                        cache_examples=False,
                    )
                    video_model = gr.Dropdown(
                        choices=list(MODELLER.keys()),
                        value=list(MODELLER.keys())[0],
                        label="Model",
                    )
                    video_esik = gr.Slider(
                        minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                        label="Güven eşiği",
                    )
                    en_az = gr.Slider(
                        minimum=1, maximum=30, value=3, step=1,
                        label="En az kaç karede görülsün",
                        info="Bir kimlik bu kadar karede görülmediyse sayıma katılmaz "
                             "(anlık yanlış alarmları eler).",
                    )
                    video_calistir = gr.Button("Takip et", variant="primary")
                    gr.Markdown(f"*Not: en fazla {MAKS_KARE} kare işlenir. "
                                f"100 kare CPU'da yaklaşık 5 saniye sürer.*")

                with gr.Column():
                    cikti_video = gr.Video(label="Takip sonucu", height=340)
                    video_ozet = gr.Markdown()
                    video_tablo = gr.Dataframe(
                        headers=["Kimlik", "Sınıf", "En yüksek güven",
                                 "Kaç karede görüldü", "İlk görüldüğü kare"],
                        label="Takip edilen nesneler",
                        wrap=True,
                    )

    # --- olaylar: düğmeye basınca ve girdi değişince çalışsın ---
    foto_girdi = [girdi_resim, model_secim, esik]
    foto_cikti = [cikti_resim, tablo, ozet_metni]
    calistir.click(tespit_et, foto_girdi, foto_cikti)
    girdi_resim.change(tespit_et, foto_girdi, foto_cikti)
    model_secim.change(tespit_et, foto_girdi, foto_cikti)
    esik.change(tespit_et, foto_girdi, foto_cikti)

    # Video ağır bir işlem: her kaydırıcı oynamasında değil, yalnızca düğmeye
    # basınca çalışsın.
    video_calistir.click(video_isle,
                         [girdi_video, video_model, video_esik, en_az],
                         [cikti_video, video_tablo, video_ozet])


if __name__ == "__main__":
    arayuz.launch()
