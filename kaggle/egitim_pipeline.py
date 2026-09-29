"""
Kaggle eğitim betiği — YOLO11n fine-tuning + yapay veri üretimi

Bu dosya, Kaggle notebook'unda parça parça çalıştırdığımız adımların
temizlenmiş ve yeniden çalıştırılabilir halidir.

KULLANIM (Kaggle notebook'unda):
    !pip install -q ultralytics
    SURUM = "v4"          # v1 | v3 | v4  → aşağıdaki tabloya bakın
    exec(open("egitim_pipeline.py").read())

Üç sürüm, tek değişkenli deney zinciri:

    | Sürüm | Gerçek foto | İHA yapıştırması | Karşı örnek | Amaç                                  |
    |-------|-------------|------------------|-------------|---------------------------------------|
    | v1    | 3.000       | 0                | 0           | Temel model                           |
    | v3    | 3.000       | 1.500            | 0           | Kestirme öğrenmeyi kırmak             |
    | v4    | 3.000       | 1.500            | 750         | Yan etkiyi gidermek (karşı örnekler)  |

Diğer her şey sabit: yolo11n, 20 epoch, batch 32, imgsz 640, random_state=42.
Böylece sonuçlardaki fark yalnızca yapay veriden kaynaklanır.

Gereksinimler: Kaggle notebook'unda GPU ve Internet açık olmalı.
"""

import os
import random
import shutil

import cv2
import pandas as pd
from ultralytics import YOLO

# =====================================================================
# AYARLAR
# =====================================================================
SURUM = globals().get("SURUM", "v4")               # dışarıdan verilmemişse v4

AYAR = {
    "v1": {"iha_yapistirma": 0,    "karsi_ornek": 0},
    "v3": {"iha_yapistirma": 1500, "karsi_ornek": 0},
    "v4": {"iha_yapistirma": 1500, "karsi_ornek": 750},
}[SURUM]

KOK   = ("/kaggle/input/datasets/umuttuygurr/aerial-uav-thermal-inferred-unified-dataset"
         "/merged_uav_dataset")
HEDEF = f"/kaggle/working/{SURUM}_verisi"
YAML  = f"/kaggle/working/data_{SURUM}.yaml"

GERCEK_TRAIN = 3000            # v1 ile aynı olmalı: karşılaştırmanın adil olması için
GERCEK_VAL   = 600
EPOCH        = 20
BATCH        = 32
IMGSZ        = 640

SINIFLAR = ["person", "car", "bicycle", "other_vehicle", "drone", "mine", "gun"]
DRONE = 4                      # drone sınıfının numarası


# =====================================================================
# 1) GERÇEK VERİ — kısayollarla (kopyalama yok, disk şişmesin)
# =====================================================================
def gercek_veriyi_hazirla():
    shutil.rmtree(HEDEF, ignore_errors=True)                  # tekrar çalıştırılabilir olsun

    for bolum, adet in [("train", GERCEK_TRAIN), ("val", GERCEK_VAL)]:
        dosyalar = sorted(os.listdir(f"{KOK}/images/{bolum}"))
        random.seed(42)                                       # her sürümde AYNI fotoğraflar seçilsin
        random.shuffle(dosyalar)

        os.makedirs(f"{HEDEF}/images/{bolum}", exist_ok=True)
        os.makedirs(f"{HEDEF}/labels/{bolum}", exist_ok=True)

        sayac = 0
        for dosya_adi in dosyalar[:adet]:
            ad = dosya_adi.rsplit(".", 1)[0]
            etiket = f"{KOK}/labels/{bolum}/{ad}.txt"
            if os.path.exists(etiket):
                os.symlink(f"{KOK}/images/{bolum}/{dosya_adi}",
                           f"{HEDEF}/images/{bolum}/{dosya_adi}")
                os.symlink(etiket, f"{HEDEF}/labels/{bolum}/{ad}.txt")
                sayac += 1
        print(f"gerçek {bolum}: {sayac}")


# =====================================================================
# 2) KESİT TOPLAMA
#    manifest.csv sayesinde yalnızca ilgili fotoğrafları açıyoruz
#    (20.000 dosyayı tek tek taramak dakikalar sürüyordu)
# =====================================================================
def kesitleri_topla(iha_hedef=400, karsi_hedef=150):
    tablo = pd.read_csv(f"{KOK}/manifest.csv")
    egitim = tablo[tablo["split"] == "train"]

    iha_kesitleri, karsi_kesitler = [], []

    def kes(satir_bilgi, istenen_siniflar, liste, sinif_ile=False, en_az=10):
        ad = satir_bilgi["output_filename"].rsplit(".", 1)[0]
        foto = cv2.imread(f"{KOK}/images/train/{ad}.png")
        if foto is None:
            return
        yukseklik, genislik = foto.shape[0], foto.shape[1]

        for satir in open(f"{KOK}/labels/train/{ad}.txt").read().splitlines():
            p = satir.split()
            sinif = int(p[0])
            if sinif not in istenen_siniflar:
                continue
            mx, my = float(p[1]) * genislik, float(p[2]) * yukseklik    # oran → piksel
            g,  y  = float(p[3]) * genislik, float(p[4]) * yukseklik
            kesit = foto[round(my - y/2):round(my + y/2),
                         round(mx - g/2):round(mx + g/2)].copy()
            if kesit.shape[0] >= en_az and kesit.shape[1] >= en_az:
                liste.append((sinif, kesit) if sinif_ile else kesit)

    # İHA kesitleri: drone içeren fotoğraflardan
    for _, satir_bilgi in egitim[egitim["drone"] > 0].sample(iha_hedef, random_state=1).iterrows():
        kes(satir_bilgi, {DRONE}, iha_kesitleri)

    # Karşı örnek kesitleri: person/car/bicycle içeren fotoğraflardan
    # en_az=24 → karşı örnek belirgin olmalı ki model "bu drone değil" diyebilsin
    if AYAR["karsi_ornek"]:
        havuz = egitim[(egitim["car"] > 0) | (egitim["person"] > 0)]
        for _, satir_bilgi in havuz.sample(karsi_hedef, random_state=2).iterrows():
            kes(satir_bilgi, {0, 1, 2}, karsi_kesitler, sinif_ile=True, en_az=24)

    print(f"İHA kesiti: {len(iha_kesitleri)} | karşı örnek kesiti: {len(karsi_kesitler)}")
    return iha_kesitleri, karsi_kesitler


# =====================================================================
# 3) YAPAY VERİ ÜRETİMİ
#    Arka plan, konum, boyut ve hangi kesit — HEPSİ rastgele.
#    Sabit kalan tek şey nesnenin kendi görüntüsü; model şekli öğrenmek
#    zorunda kalsın diye.
# =====================================================================
def yapistir_ve_kaydet(arka_ad, eklenecekler, yeni_ad):
    """eklenecekler: (sınıf_no, kesit) listesi. Arka planın kendi etiketleri KORUNUR."""
    foto = cv2.imread(f"{HEDEF}/images/train/{arka_ad}.png")
    if foto is None:
        return False
    yukseklik, genislik = foto.shape[0], foto.shape[1]

    satirlar = open(f"{HEDEF}/labels/train/{arka_ad}.txt").read().splitlines()
    yapay = foto.copy()

    for sinif, kesit in eklenecekler:
        olcek = random.uniform(0.6, 1.8)
        yeni_g = max(12, int(kesit.shape[1] * olcek))
        yeni_y = max(12, int(kesit.shape[0] * olcek))
        if yeni_g >= genislik or yeni_y >= yukseklik:
            continue
        k = cv2.resize(kesit, (yeni_g, yeni_y))               # resize (genişlik, yükseklik) ister

        x = random.randint(0, genislik - yeni_g - 1)
        y = random.randint(0, yukseklik - yeni_y - 1)
        yapay[y:y+yeni_y, x:x+yeni_g] = k

        # piksel → oran (etiketi biz yazıyoruz; yapıştırma koordinatı biliniyor)
        satirlar.append(f"{sinif} {(x + yeni_g/2)/genislik:.6f} {(y + yeni_y/2)/yukseklik:.6f} "
                        f"{yeni_g/genislik:.6f} {yeni_y/yukseklik:.6f}")

    cv2.imwrite(f"{HEDEF}/images/train/{yeni_ad}.png", yapay)  # kısayol değil, gerçek dosya
    open(f"{HEDEF}/labels/train/{yeni_ad}.txt", "w").write("\n".join(satirlar) + "\n")
    return True


def yapay_veri_uret(iha_kesitleri, karsi_kesitler):
    gokyuzu_olmayan, gokyuzu = [], []
    for dosya_adi in os.listdir(f"{HEDEF}/images/train"):
        ad = dosya_adi.rsplit(".", 1)[0]
        (gokyuzu if dosya_adi.startswith("01_uav2uav") else gokyuzu_olmayan).append(ad)

    # --- İHA'ları gökyüzü OLMAYAN sahnelere yapıştır ---
    random.seed(7)
    sayac = 0
    for i in range(AYAR["iha_yapistirma"]):
        arka = random.choice(gokyuzu_olmayan)
        eklenecek = [(DRONE, random.choice(iha_kesitleri)) for _ in range(random.randint(1, 3))]
        sayac += yapistir_ve_kaydet(arka, eklenecek, f"yapay_iha_{i:05d}")
    if AYAR["iha_yapistirma"]:
        print("İHA yapıştırması:", sayac)

    # --- KARŞI ÖRNEK: araba/insan/bisikleti GÖKYÜZÜNE yapıştır, kendi sınıfıyla etiketle ---
    random.seed(8)
    sayac = 0
    for i in range(AYAR["karsi_ornek"]):
        arka = random.choice(gokyuzu)
        eklenecek = [random.choice(karsi_kesitler) for _ in range(random.randint(1, 2))]
        sayac += yapistir_ve_kaydet(arka, eklenecek, f"yapay_karsi_{i:05d}")
    if AYAR["karsi_ornek"]:
        print("karşı örnek:", sayac)

    print("toplam eğitim fotoğrafı:", len(os.listdir(f"{HEDEF}/images/train")))


# =====================================================================
# 4) data.yaml — YOLO ile veri klasörü arasındaki köprü
#    Etiket klasörü yazılmaz: YOLO yoldaki "images" kelimesini "labels"
#    ile değiştirip kendisi bulur.
# =====================================================================
def yaml_yaz():
    isimler = "\n".join(f"  {i}: {ad}" for i, ad in enumerate(SINIFLAR))
    open(YAML, "w").write(
        f"path: {HEDEF}\ntrain: images/train\nval: images/val\n\n"
        f"nc: {len(SINIFLAR)}\nnames:\n{isimler}\n"
    )
    print("yazıldı:", YAML)


# =====================================================================
# 5) EĞİTİM — hazır ağırlıklardan devam (transfer öğrenme / fine-tuning)
#    Mimari değişmiyor; yalnızca çıkış katmanı 80 → 7 sınıfa iniyor.
# =====================================================================
def egit():
    model = YOLO("yolo11n.pt")
    model.train(
        data=YAML,
        epochs=EPOCH,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,                                  # GPU
        project="/kaggle/working/egitim",
        name=f"deneme_{SURUM}",
    )


if __name__ == "__main__":
    print(f"=== SÜRÜM: {SURUM} | {AYAR} ===")
    gercek_veriyi_hazirla()

    iha_kesitleri, karsi_kesitler = [], []
    if AYAR["iha_yapistirma"] or AYAR["karsi_ornek"]:
        iha_kesitleri, karsi_kesitler = kesitleri_topla()
        yapay_veri_uret(iha_kesitleri, karsi_kesitler)

    yaml_yaz()
    egit()
