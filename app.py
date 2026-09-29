"""
İHA / termal görüntü nesne tespiti — demo arayüzü

Kullanıcı bir fotoğraf yükler, seçilen model nesneleri tespit eder,
kutular çizilir ve bulunan her nesne sınıf + güven skoruyla listelenir.

Çalıştırmak için:  python app.py
"""

import cv2
import numpy as np
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

_yuklu = {}                                    # aynı modeli tekrar tekrar yüklememek için


def model_getir(ad):
    """Model dosyasını yükler ve bellekte tutar (ilk çağrıda yükler)."""
    if ad not in _yuklu:
        _yuklu[ad] = YOLO(MODELLER[ad])
    return _yuklu[ad]


# =====================================================================
# 2) TESPİT FONKSİYONU
#    Gradio bu fonksiyonu çağırır: girdileri alır, çıktıları döndürür.
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
# 3) ARAYÜZ
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

with gr.Blocks(title="İHA Tespit Demo") as arayuz:
    gr.Markdown(ACIKLAMA)

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

    # düğmeye basınca ve görüntü değişince çalışsın
    calistir.click(tespit_et, [girdi_resim, model_secim, esik], [cikti_resim, tablo, ozet_metni])
    girdi_resim.change(tespit_et, [girdi_resim, model_secim, esik], [cikti_resim, tablo, ozet_metni])
    model_secim.change(tespit_et, [girdi_resim, model_secim, esik], [cikti_resim, tablo, ozet_metni])
    esik.change(tespit_et, [girdi_resim, model_secim, esik], [cikti_resim, tablo, ozet_metni])


if __name__ == "__main__":
    arayuz.launch()
