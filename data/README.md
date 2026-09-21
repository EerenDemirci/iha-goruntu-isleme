# Veri

**Veri seti:** [Aerial UAV Thermal – Inferred Unified Dataset](https://www.kaggle.com/datasets/umuttuygurr/aerial-uav-thermal-inferred-unified-dataset) (Kaggle)

| | |
|---|---|
| Görüntü sayısı | 27.925 (kuşbakışı, İHA, termal/kızılötesi) |
| Nesne sayısı | 90.448 |
| Sınıflar (7) | Person, Car, Bicycle, Other Vehicle, Drone/UAV, Mine, Gun |
| Etiket formatı | YOLO (`.txt`) |
| Bölümler | train / val / test |
| Boyut | 13,2 GB |
| Lisans | CC0: Public Domain |

**Kullanım planı**
- Veri setinin tamamı boyutu nedeniyle yerele indirilmez. Bu klasöre yalnızca öğrenmek için küçük bir alt küme (birkaç yüz görüntü) konur.
- Model eğitimi Kaggle Notebook üzerinde, veri seti doğrudan bağlanarak GPU ile yapılır.
- Bu klasörün içeriği git'e eklenmez, yalnızca bu README repoda yer alır.
