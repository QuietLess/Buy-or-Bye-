# Buy or Bye — Kapsamlı Proje, Deney ve Model Raporu

> Son güncelleme: 5 Ağustos 2026  
> Proje türü: Dengesiz ikili sınıflandırma, olasılık kalibrasyonu ve iş segmentasyonu  
> Final model: **Calibrated RF + Engineered LightGBM Ensemble**  
> Ensemble ağırlıkları: **%80 Random Forest + %20 Engineered LightGBM**  
> Final karar eşiği: **0.25**

## İçindekiler

1. [Yönetici özeti](#1-yönetici-özeti)
2. [İş problemi ve model çıktısı](#2-iş-problemi-ve-model-çıktısı)
3. [Veri seti](#3-veri-seti)
4. [Özellik sözlüğü](#4-özellik-sözlüğü)
5. [Keşifsel veri analizi ve veri kalitesi](#5-keşifsel-veri-analizi-ve-veri-kalitesi)
6. [Eğitim ve değerlendirme protokolü](#6-eğitim-ve-değerlendirme-protokolü)
7. [Metriklerin anlamı](#7-metriklerin-anlamı)
8. [Baseline model deneyi](#8-baseline-model-deneyi)
9. [Hyperparameter tuning](#9-hyperparameter-tuning)
10. [Kalibrasyon ve threshold seçimi](#10-kalibrasyon-ve-threshold-seçimi)
11. [Feature engineering](#11-feature-engineering)
12. [Overfitting ve robustness değerlendirmesi](#12-overfitting-ve-robustness-değerlendirmesi)
13. [Random Forest + LightGBM ensemble deneyi](#13-random-forest--lightgbm-ensemble-deneyi)
14. [Final model ve test sonuçları](#14-final-model-ve-test-sonuçları)
15. [SHAP açıklanabilirlik](#15-shap-açıklanabilirlik)
16. [Olasılık segmentasyonu ve iş aksiyonları](#16-olasılık-segmentasyonu-ve-iş-aksiyonları)
17. [Streamlit demo](#17-streamlit-demo)
18. [PageValues ve data leakage değerlendirmesi](#18-pagevalues-ve-data-leakage-değerlendirmesi)
19. [Bilinen sınırlamalar](#19-bilinen-sınırlamalar)
20. [Önerilen sonraki deneyler](#20-önerilen-sonraki-deneyler)
21. [Çalıştırma komutları ve artifact haritası](#21-çalıştırma-komutları-ve-artifact-haritası)

---

## 1. Yönetici özeti

Buy or Bye, bir e-ticaret oturumunun satın alma ile sonuçlanma olasılığını tahmin eden uçtan uca bir makine öğrenmesi projesidir. Çözüm yalnız `0/1` etiketi üretmez; kalibre edilmiş `predict_proba()` çıktısını kullanarak her oturumu `0.00–1.00` aralığında skorlar, üç niyet segmentinden birine atar ve segmente özel iş aksiyonu önerir.

Başlıca sonuçlar:

- Ham veri **12.330 oturumdur**; splitten önce 125 exact duplicate kaldırılmış ve modelleme **12.205 tekil oturumla** yapılmıştır.
- Pozitif sınıf oranı **%15,63** olduğu için problem dengesizdir.
- Model seçiminin ana metriği accuracy değil **PR-AUC** olmuştur.
- Baseline LightGBM test PR-AUC değeri **0.7445** idi.
- Tuning sonrasında kalibre LightGBM test PR-AUC değeri **0.7516** oldu.
- 27 yeni davranış özelliği ve yeniden tuning sonrasında Engineered LightGBM CV PR-AUC **0.7562** oldu.
- Validation ağırlık deneyinde `%80 RF + %20 Engineered LightGBM` ensemble en iyi sonucu verdi ve önceden tanımlı kuralla final artifact oldu.
- Final ensemble test sonuçları: **ROC-AUC 0.9320**, **PR-AUC 0.7448**, **F1 0.6736**, **recall 0.7902**, **precision 0.5870**.
- Ensemble calibration test Brier skorunu **0.0860 → 0.0696**, log loss değerini **0.2805 → 0.2270** iyileştirdi.
- Engineered LightGBM beş seed üzerinde PR-AUC **0.7662 ± 0.0077** üretti.
- Zaman proxy testinde PR-AUC **0.6655** seviyesine düştü; dönem kayması riski vardır.
- Üç segmentin test dönüşüm oranları sırasıyla **%2,7**, **%19,7** ve **%58,7** oldu.
- `PageValues` çıkarıldığında engineered CV PR-AUC **0.7562 → 0.3797** düştü.

Final karar: Önceden belirlenen validation protokolüne göre final artifact **Calibrated RF + Engineered LightGBM Ensemble**’dır. Tekil Engineered LightGBM test PR-AUC’si `0.7507` ile ensemble’ın `0.7448` sonucundan yüksektir; test seçim için kullanılmadığından bu gözlem final kararı geriye dönük değiştirmemiştir. Her iki yapı da `PageValues` kullanan analytics-zengin senaryoyu temsil eder.

---

## 2. İş problemi ve model çıktısı

### 2.1 Amaç

Bir ziyaretçinin mevcut e-ticaret oturumunun satın alma ile bitme ihtimalini tahmin etmek ve aşağıdaki kararları desteklemek:

- Hangi oturumlara checkout desteği verilmeli?
- Hangi kullanıcılar düşük maliyetli yeniden hedeflemeye alınmalı?
- Hangi oturumlarda indirim vermeden organik dönüşüm beklenebilir?
- Pazarlama ve ürün ekipleri hangi davranış sinyallerine odaklanmalı?

### 2.2 Model çıktıları

Model üç düzeyde çıktı üretir:

1. **Sürekli olasılık:** `predict_proba()[:, 1]`, `0.00–1.00`.
2. **Binary sinyal:** olasılık `0.25` veya üzerindeyse satın alma sinyali.
3. **İş segmenti:** düşük, değerlendirme veya yüksek niyet.

`0.25` eşiği keyfi seçilmemiştir. Ensemble threshold validation yarısında yanlış negatif maliyeti `2`, yanlış pozitif maliyeti `1` kabul edilerek minimum iş maliyetini veren threshold seçilmiştir.

---

## 3. Veri seti

### 3.1 Kaynak

- Ad: **Online Shoppers Purchasing Intention Dataset**
- Sağlayıcı: UCI Machine Learning Repository
- DOI: `10.24432/C5F88Q`
- Lisans: CC BY 4.0
- Gözlem birimi: Tek e-ticaret oturumu
- Ham satır sayısı: **12.330**
- Deduplicate modelleme satırı: **12.205**
- Girdi sayısı: **17**
- Hedef: `Revenue`

### 3.2 Hedef dağılımı

| Revenue | Oturum | Oran |
|---|---:|---:|
| `False` / satın almama | 10.297 | %84,37 |
| `True` / satın alma | 1.908 | %15,63 |

Pozitif/negatif oranı yaklaşık `1:5,40` seviyesindedir. Bu nedenle yalnız accuracy kullanmak yanıltıcıdır; tüm oturumlara “satın almama” demek bile yaklaşık `%84,4` accuracy verir.

---

## 4. Özellik sözlüğü

### 4.1 Sayısal özellikler

| Teknik ad | Kullanıcı dostu ad | Açıklama |
|---|---|---|
| `Administrative` | Hesap ve işlem adımı görüntüleme sayısı | Veri setinin idari/operasyonel olarak grupladığı hesap ve işlem sayfalarının görüntülenme sayısı. |
| `Administrative_Duration` | Hesap ve işlem adımlarında geçirilen süre | Bu sayfalardaki toplam süre, saniye. |
| `Informational` | Yardım ve bilgi içeriği görüntüleme sayısı | Yardım, politika, iletişim ve bilgilendirme türü içerik görüntüleme sayısı. |
| `Informational_Duration` | Yardım ve bilgi içeriklerinde geçirilen süre | Bilgilendirici sayfalardaki toplam süre, saniye. |
| `ProductRelated` | Ürün inceleme sayfası görüntüleme sayısı | Ürün listeleme, detay ve inceleme sayfası görüntüleme sayısı. |
| `ProductRelated_Duration` | Ürün inceleme sayfalarında geçirilen süre | Ürünle ilgili sayfalardaki toplam süre, saniye. |
| `BounceRates` | Tek sayfadan ayrılma oranı | Başka bir sayfaya geçmeden ayrılma eğilimini gösteren analytics oranı. |
| `ExitRates` | Sayfa sonrası siteden çıkış oranı | Sayfanın oturumdaki son sayfa olma eğilimi. |
| `PageValues` | Sayfaların dönüşüm değeri | Analytics sisteminin ziyaret edilen sayfaların dönüşüme parasal katkısını özetleyen değer. Ürün fiyatı veya olasılık değildir. |
| `SpecialDay` | Kampanya/özel güne yakınlık | Özel güne yakınlığı gösteren `0–1` skoru. |

### 4.2 Kategorik özellikler

| Teknik ad | Kullanıcı dostu ad | Değerler / açıklama |
|---|---|---|
| `OperatingSystems` | İşletim sistemi grubu | Anonim `1–8` kodları. |
| `Browser` | Tarayıcı grubu | Anonim `1–13` kodları. |
| `Region` | Bölge grubu | Anonim `1–9` kodları. |
| `TrafficType` | Ziyaret kaynağı grubu | Anonim `1–20` trafik kaynağı kodları. |
| `VisitorType` | Ziyaretçi geçmişi | `Returning_Visitor`, `New_Visitor`, `Other`. |
| `Weekend` | Hafta sonu | Boolean. |
| `Month` | Ziyaret ayı | Feb, Mar, May, June, Jul, Aug, Sep, Oct, Nov, Dec. |

Kategori kodlarının gerçek işletim sistemi, tarayıcı veya kampanya isimleri veri setinde anonimleştirilmiştir. Üretim sisteminde bu değerler manuel değil analytics entegrasyonundan gelmelidir.

---

## 5. Keşifsel veri analizi ve veri kalitesi

### 5.1 Sayısal özet

| Özellik | Min | Ortalama | Std | %25 | Medyan | %75 | Maksimum |
|---|---:|---:|---:|---:|---:|---:|---:|
| Administrative | 0 | 2,339 | 3,330 | 0 | 1 | 4 | 27 |
| Administrative_Duration | 0 | 81,646 | 177,492 | 0 | 9,0 | 94,7 | 3.398,75 |
| Informational | 0 | 0,509 | 1,276 | 0 | 0 | 0 | 24 |
| Informational_Duration | 0 | 34,825 | 141,425 | 0 | 0 | 0 | 2.549,375 |
| ProductRelated | 0 | 32,046 | 44,594 | 8 | 18 | 38 | 705 |
| ProductRelated_Duration | 0 | 1.206,982 | 1.919,601 | 193,0 | 608,943 | 1.477,155 | 63.973,522 |
| BounceRates | 0 | 0,02037 | 0,04526 | 0 | 0,002899 | 0,016667 | 0,20 |
| ExitRates | 0 | 0,04147 | 0,04616 | 0,014231 | 0,025000 | 0,048529 | 0,20 |
| PageValues | 0 | 5,950 | 18,654 | 0 | 0 | 0 | 361,764 |
| SpecialDay | 0 | 0,06194 | 0,19967 | 0 | 0 | 0 | 1 |

Süre ve sayfa adedi değişkenleri belirgin biçimde sağa çarpıktır. Bu nedenle feature engineering aşamasında log dönüşümleri ve sayfa başına süre oranları eklenmiştir.

### 5.2 Eksik ve mükerrer kayıtlar

| Kontrol | Sonuç |
|---|---:|
| Eksik hücre | 0 |
| Ham veride tam mükerrer satır | 125 |
| Modellemeye giren tekil satır | 12.205 |
| Temiz veride mükerrer satır | 0 |

`load_data()` ham CSV’yi korur fakat doğrulamadan hemen sonra `drop_duplicates()` uygular. Deduplication splitten önce yapıldığı için yeni train–validation ve train–test exact overlap sayısı **0**’dır. 125 tekrarın tamamı negatif sınıftaydı; pozitif satır sayısı 1.908 olarak kaldı. Tüm model, calibration, threshold, ensemble, segment, SHAP, grafik, notebook ve sunum artifact’leri temiz veriyle yeniden üretildi.

### 5.3 Ana kategorik dağılımlar

#### Ziyaretçi tipi

| Ziyaretçi tipi | Oturum | Dönüşüm oranı |
|---|---:|---:|
| Returning Visitor | 10.431 | %14,09 |
| New Visitor | 1.693 | %24,93 |
| Other | 81 | %19,75 |

#### Hafta sonu

| Dönem | Oturum | Dönüşüm oranı |
|---|---:|---:|
| Hafta içi | 9.346 | %15,08 |
| Hafta sonu | 2.859 | %17,45 |

#### Ay

| Ay | Oturum | Dönüşüm oranı |
|---|---:|---:|
| Feb | 181 | %1,66 |
| Mar | 1.860 | %10,32 |
| May | 3.329 | %10,96 |
| June | 285 | %10,18 |
| Jul | 432 | %15,28 |
| Aug | 433 | %17,55 |
| Sep | 448 | %19,20 |
| Oct | 549 | %20,95 |
| Nov | 2.982 | %25,49 |
| Dec | 1.706 | %12,66 |

Kasım dönüşüm oranı ve hacmi yüksek, Şubat ise çok düşüktür. Bu nedenle `Month` hem one-hot hem döngüsel sin/cos özelliklerle temsil edilmiştir.

### 5.4 Hedef ile sayısal korelasyonlar

| Özellik | Revenue korelasyonu |
|---|---:|
| PageValues | +0,4919 |
| ProductRelated | +0,1560 |
| ProductRelated_Duration | +0,1501 |
| Administrative | +0,1363 |
| Informational | +0,0936 |
| Administrative_Duration | +0,0918 |
| Informational_Duration | +0,0694 |
| SpecialDay | -0,0836 |
| BounceRates | -0,1451 |
| ExitRates | -0,2043 |

Korelasyon doğrusal ilişkiyi ölçer ve nedensellik göstermez. `PageValues` açık ara en güçlü tekil doğrusal ilişkiye sahiptir.

### 5.5 PageValues dağılımı

- `PageValues > 0` olan oturum: **2.730 (%22,37)**
- Tüm veride medyan: **0**
- Tüm veride ortalama: **5,9496**
- Sıfır-dışı medyan: **16,6556**
- Sıfır-dışı %25 / %75: **7,2131 / 34,7802**
- Maksimum: **361,7637**

Dağılımın çoğunluğu sıfırdır. Bu nedenle `0`, “eksik” anlamına gelmez; gerçek bir değer olabilir. Eksik Page Value’yu otomatik olarak `0` veya medyanla doldurmak semantik olarak doğru değildir.

---

## 6. Eğitim ve değerlendirme protokolü

### 6.1 Split

Deterministik ve stratified `70/15/15` split, `random_state=42` ile oluşturulmuştur.

| Bölüm | Satır | Negatif | Pozitif | Pozitif oranı |
|---|---:|---:|---:|---:|
| Train | 8.543 | 7.207 | 1.336 | %15,64 |
| Validation | 1.831 | 1.545 | 286 | %15,62 |
| Test | 1.831 | 1.545 | 286 | %15,62 |

Kullanım politikası:

- Train: model fitting, 5-fold tuning ve OOF calibration.
- Validation: threshold ve iş maliyeti seçimi.
- Test: yalnız final raporlama.

Feature engineering deneyinde terfi kararı test sonucuna göre değil train CV PR-AUC’ye göre verilmiştir.

### 6.2 Preprocessing

Tüm dönüşümler scikit-learn pipeline içinde fit edilmiştir:

- Sayısal eksikler: medyan imputation.
- Sayısal ölçekleme: `StandardScaler`.
- Kategorik eksikler: en sık kategori.
- Kategorik encoding: `OneHotEncoder(handle_unknown="ignore")`.
- Yeni kategoriler inference sırasında hata üretmez.
- Feature engineering pipeline içinde olduğu için validation/test verisinden istatistik öğrenilmez.

### 6.3 Class imbalance yaklaşımı

Sınıf dengesizliği şu şekilde ele alınmıştır:

- Logistic Regression: `class_weight="balanced"`.
- Random Forest: tuning içinde `balanced` ve `balanced_subsample`.
- LightGBM: `class_weight="balanced"` taban ayarı.
- Model seçimi: PR-AUC.
- Stratified split ve StratifiedKFold.
- Threshold, varsayılan `0.50` yerine iş maliyetine göre optimize edildi.

SMOTE uygulanmadı. Ağaç modellerinde class weighting, kalibrasyon ve threshold tuning’in daha kontrollü olması tercih edildi; sentetik oturum üretmenin kategorik ve analytics ilişkilerini bozma riski vardı.

---

## 7. Metriklerin anlamı

| Metrik | Anlamı | Bu projede kullanımı |
|---|---|---|
| Accuracy | Tüm doğru tahminlerin oranı | Dengesizlik nedeniyle ikincil. |
| Precision | Satın alır denenlerin ne kadarı gerçekten aldı? | Kampanya maliyeti ve gereksiz aksiyonu ölçer. |
| Recall | Gerçek alıcıların ne kadarı yakalandı? | Kaçırılan satın alma fırsatını ölçer. |
| F1 | Precision ve recall harmonik ortalaması | Tek threshold skoru olarak dengeli özet. |
| ROC-AUC | Pozitifleri negatiflerin önüne sıralama gücü | Genel ranking ölçüsü; dengesizlikte iyimser görünebilir. |
| PR-AUC / Average Precision | Pozitif sınıf precision-recall kalitesi | Ana model seçme metriği. Rastgele referans yaklaşık pozitif oranı `%15,47`dir. |
| Brier score | Olasılık ile gerçek sonuç arasındaki ortalama karesel hata | Düşük daha iyi; calibration kalitesini ölçer. |
| Log loss | Yanlış ve aşırı emin olasılıkları cezalandırır | Düşük daha iyi. |

Confusion matrix sırası `[[TN, FP], [FN, TP]]` şeklindedir.

---

## 8. Baseline model deneyi

İlk aşamada Logistic Regression, Random Forest ve LightGBM basit pipeline’larla karşılaştırılmıştır.

### 8.1 Validation

| Model | ROC-AUC | PR-AUC | Optimize eşik | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0,9021 | 0,6831 | 0,61 | 0,6174 | 0,6713 | 0,6432 | 0,8837 |
| Random Forest | 0,9189 | 0,7114 | 0,36 | 0,5728 | **0,8112** | 0,6715 | 0,8760 |
| LightGBM | **0,9278** | **0,7369** | 0,70 | **0,6866** | 0,6818 | **0,6842** | **0,9017** |

Baseline aşamasında LightGBM en iyi PR-AUC ve F1 değerini verdi.

### 8.2 Baseline LightGBM test sonucu

| Metrik | Değer |
|---|---:|
| ROC-AUC | 0,9301 |
| PR-AUC | 0,7445 |
| Eşik | 0,70 |
| Precision | 0,6655 |
| Recall | 0,6748 |
| F1 | 0,6701 |
| Accuracy | 0,8962 |
| Confusion matrix | `[[1448, 97], [93, 193]]` |

---

## 9. Hyperparameter tuning

### 9.1 Protokol

- CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Ana scoring: `average_precision` / PR-AUC
- Logistic Regression: GridSearch, 12 kombinasyon.
- Random Forest: RandomizedSearch, 8 örnek.
- LightGBM: RandomizedSearch, 16 örnek.
- Train skorları overfitting gözlemi için kaydedildi.

### 9.2 Arama uzayları

#### Logistic Regression

- `C`: 0.03, 0.1, 0.3, 1, 3, 10
- penalty: L1, L2
- solver: liblinear

#### Random Forest

- n_estimators: 150, 250, 350
- max_depth: 8, 12, 16, 20, None
- min_samples_leaf: 1, 2, 4, 8, 12
- min_samples_split: 2, 5, 10, 20
- max_features: sqrt, 0.5, 0.8
- class_weight: balanced, balanced_subsample

#### LightGBM

- n_estimators: 200, 350, 500, 700, 900
- learning_rate: 0.02, 0.03, 0.05, 0.08
- max_depth: 4, 5, 6, 7, 8, -1
- num_leaves: 15, 23, 31, 47, 63
- min_child_samples: 10, 20, 35, 50, 80
- subsample: 0.7, 0.85, 1.0
- colsample_bytree: 0.65, 0.8, 0.95, 1.0
- reg_alpha: 0, 0.05, 0.2, 0.5, 1.0
- reg_lambda: 0, 0.1, 0.5, 1.0, 2.0

### 9.3 Tuning sonuçları

| Model | CV train PR-AUC | CV validation PR-AUC | Std | Gözlem |
|---|---:|---:|---:|---|
| Logistic Regression | 0,6567 | 0,6554 | 0,0264 | Düşük varyans, düşük kapasite. |
| Random Forest | 0,9040 | 0,7522 | 0,0075 | Belirgin train-CV farkı. |
| LightGBM | 0,8343 | **0,7549** | 0,0079 | En iyi CV ve RF’den daha düşük train-CV farkı. |

Seçilen baseline tuned model LightGBM oldu.

### 9.4 En iyi parametreler

#### Logistic Regression

```text
C=0.03, penalty=l1, solver=liblinear
```

#### Random Forest

```text
n_estimators=350
max_depth=None
min_samples_leaf=8
min_samples_split=10
max_features=0.5
class_weight=balanced_subsample
```

#### LightGBM

```text
n_estimators=200
learning_rate=0.03
max_depth=5
num_leaves=31
min_child_samples=50
subsample=0.7
subsample_freq=1
colsample_bytree=0.95
reg_alpha=0.2
reg_lambda=0.0
```

### 9.5 Tuned LightGBM test sonucu

| Metrik | Değer |
|---|---:|
| ROC-AUC | 0,9359 |
| PR-AUC | 0,7516 |
| Eşik | 0,46 |
| Precision | 0,7080 |
| Recall | 0,6783 |
| F1 | 0,6929 |
| Accuracy | 0,9061 |
| Confusion matrix | `[[1465, 80], [92, 194]]` |

---

## 10. Kalibrasyon ve threshold seçimi

### 10.1 OOF sigmoid kalibrasyonu

Final estimator için train üzerinde 5-fold out-of-fold olasılıklar üretildi. Bu OOF skorlarla Logistic Regression tabanlı sigmoid/Platt calibrator eğitildi. Sonrasında estimator tüm train verisiyle fit edildi; calibrator test etiketlerini görmedi.

### 10.2 Kalibrasyon sonuçları

#### Tuned LightGBM

| Veri | Brier ham | Brier kalibre | Log loss ham | Log loss kalibre |
|---|---:|---:|---:|---:|
| Validation | 0,0991 | 0,0692 | 0,3169 | 0,2256 |
| Test | 0,0982 | 0,0683 | 0,3156 | 0,2221 |

#### Engineered LightGBM adayı

| Veri | Brier ham | Brier kalibre | Log loss ham | Log loss kalibre |
|---|---:|---:|---:|---:|
| Validation | 0,1024 | 0,0709 | 0,3336 | 0,2305 |
| Test | 0,1014 | **0,0689** | 0,3323 | **0,2242** |

#### Final ensemble

| Veri | Brier ham | Brier kalibre | Log loss ham | Log loss kalibre |
|---|---:|---:|---:|---:|
| Test | 0,0860 | **0,0696** | 0,2805 | **0,2270** |

Her iki calibration metriğinin de düşmesi, `predict_proba()` skorlarının karar ve segmentasyon için daha güvenilir hale geldiğini gösterir.

### 10.3 Threshold maliyeti

Kabul edilen maliyetler:

- False positive: `1`
- False negative: `2`

Final validation seçiminde:

| Eşik | FP | FN | İş maliyeti | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| **0,25** | 59 | 36 | **131** | 0,6446 | 0,7483 | 0,6926 |

Final ensemble threshold-validation yarısında F1-optimal ve maliyet-optimal eşik aynı noktada, `0.25`, birleşmiştir. Engineered LightGBM’in kendi validation eşiği `0.33`, tuned baseline LightGBM’in eşiği `0.46` idi.

---

## 11. Feature engineering

### 11.1 Amaç

Ham sayfa sayıları ve sürelerden yalnız hacim değil, yoğunluk, oran, etkileşim, retention ve dönemsel yapı üretmek amaçlandı. Tüm özellikler hedef kullanılmadan, deterministic olarak pipeline içinde hesaplandı.

### 11.2 Eklenen 27 özellik

#### Toplamlar ve sayfa başına yoğunluk

| Özellik | Formül / anlam |
|---|---|
| `TotalPages` | Administrative + Informational + ProductRelated |
| `TotalDuration` | Üç sayfa grubunun toplam süresi |
| `AdminDurationPerPage` | Administrative_Duration / Administrative |
| `InfoDurationPerPage` | Informational_Duration / Informational |
| `ProductDurationPerPage` | ProductRelated_Duration / ProductRelated |
| `DurationPerPage` | TotalDuration / TotalPages |

Sıfıra bölme durumları güvenli biçimde `0` ile ele alınmıştır.

#### Sayfa kompozisyonu

| Özellik | Formül / anlam |
|---|---|
| `ProductPageShare` | ProductRelated / TotalPages |
| `AdminPageShare` | Administrative / TotalPages |
| `InfoPageShare` | Informational / TotalPages |

#### Retention ve engagement

| Özellik | Formül / anlam |
|---|---|
| `ExitBounceGap` | ExitRates − BounceRates |
| `RetentionScore` | 1 − (BounceRates + ExitRates) / 2 |
| `EngagementScore` | log(1 + TotalDuration) × (1 − Bounce) × (1 − Exit) |
| `ProductEngagement` | log(1 + ProductDuration) × ProductPageShare |

#### Log dönüşümleri

| Özellik | Kaynak |
|---|---|
| `LogAdministrativeDuration` | log1p(Administrative_Duration) |
| `LogInformationalDuration` | log1p(Informational_Duration) |
| `LogProductDuration` | log1p(ProductRelated_Duration) |
| `LogTotalDuration` | log1p(TotalDuration) |
| `LogProductPages` | log1p(ProductRelated) |
| `LogPageValues` | log1p(PageValues) |

#### Dönemsellik ve binary sinyaller

| Özellik | Anlam |
|---|---|
| `MonthSin`, `MonthCos` | Ayın döngüsel sinüs/kosinüs temsili |
| `HasAdministrative` | En az bir hesap/işlem sayfası var mı? |
| `HasInformational` | En az bir bilgi sayfası var mı? |
| `HasPageValue` | PageValues > 0 mı? |

#### Etkileşimler

| Özellik | Anlam |
|---|---|
| `WeekendSpecialDay` | Weekend × SpecialDay |
| `ReturningWeekend` | Returning Visitor × Weekend |
| `PageValuePerProductPage` | PageValues / ProductRelated |

### 11.3 Sabit parametreli feature karşılaştırması

| Yapı | CV PR-AUC |
|---|---:|
| Baseline Random Forest | 0,7522 |
| Engineered Random Forest | 0,7469 ± 0,0076 |
| Baseline LightGBM | 0,7549 |
| Engineered LightGBM | **0,7559 ± 0,0098** |

Feature engineering RF’ye doğrudan fayda sağlamadı; LightGBM’de küçük bir artış üretti. Bu nedenle engineered modeller ayrıca tune edildi.

### 11.4 Engineered tuning sonuçları

| Model | CV train PR-AUC | CV validation PR-AUC | Std |
|---|---:|---:|---:|
| Engineered Random Forest | 0,8837 | 0,7529 | 0,0077 |
| Engineered LightGBM | 0,8614 | **0,7562** | 0,0080 |

Terfi kuralı: engineered model train 5-fold CV PR-AUC’de baseline en iyi skoru aşarsa promote edilir. Engineered LightGBM `0.7562 > 0.7549` olduğu için ensemble öncesi final adaya terfi etti. Artış yalnız `0.0013` civarında olduğundan mütevazıdır.

### 11.5 Final LightGBM parametreleri

```text
learning_rate=0.02
max_depth=6
num_leaves=47
min_child_samples=35
subsample=0.85
subsample_freq=1
colsample_bytree=0.85
reg_alpha=0.10
reg_lambda=0.70
```

Arama sırasında `n_estimators=200` seçildi. Ayrı early-stopping çalışmasında maksimum 1.500 tur ve 75 tur patience ile en iyi iterasyon **115** bulundu; estimator bu iterasyon sayısıyla fit edildi.

---

## 12. Overfitting ve robustness değerlendirmesi

### 12.1 Train-CV farkları

| Model | Train PR-AUC | CV PR-AUC | Fark |
|---|---:|---:|---:|
| Tuned Random Forest | 0,9040 | 0,7522 | 0,1518 |
| Tuned LightGBM | 0,8343 | 0,7549 | 0,0795 |
| Engineered Random Forest | 0,8837 | 0,7529 | 0,1308 |
| Engineered LightGBM | 0,8614 | 0,7562 | 0,1052 |

Random Forest ailesinde daha yüksek overfitting işareti vardır. Engineered LightGBM’de de train-CV farkı bulunur; model tamamen overfitting-free değildir. Buna rağmen:

- CV fold standart sapması düşüktür (`0.0080`).
- Bağımsız engineered test PR-AUC (`0.7507`) CV skoruna yakındır.
- Beş seed sonucu stabildir.
- Regularization, subsampling, column sampling, min child samples ve early stopping kullanılmıştır.

Sonuç: Ağır ve kontrolsüz overfitting kanıtı yoktur; fakat performansın PageValues’a bağımlılığı ve zaman proxy düşüşü genelleme riskini büyütmektedir.

### 12.2 Beş seed stabilitesi

| Seed | PR-AUC | ROC-AUC |
|---:|---:|---:|
| 7 | 0,7735 | 0,9439 |
| 21 | 0,7585 | 0,9332 |
| 42 | 0,7558 | 0,9366 |
| 73 | 0,7748 | 0,9378 |
| 101 | 0,7684 | 0,9389 |
| **Ortalama ± std** | **0,7662 ± 0,0077** | **0,9381 ± 0,0035** |

### 12.3 Temporal proxy

Gerçek timestamp/yıl alanı bulunmadığı için yaklaşık zaman testi uygulandı:

- Train: Feb–Oct, 7.517 satır
- Holdout: Nov–Dec, 4.688 satır
- Holdout pozitif oranı: %20,82
- PR-AUC: **0,6655**
- ROC-AUC: **0,8308**

Random split engineered test PR-AUC `0.7507` iken dönem proxy skorunun `0.6655` olması, ay dağılımı ve müşteri davranışının zamanla değişebildiğini gösterir. Gerçek timestamp ile rolling/forward validation gereklidir.

---

## 13. Random Forest + LightGBM ensemble deneyi

### 13.1 Protokol

- Her iki bileşen train üzerinde 5-fold OOF skorlarla ayrı kalibre edildi.
- Validation ikiye bölündü.
- İlk yarıda soft-voting ağırlığı seçildi.
- İkinci yarıda threshold seçildi.
- Test yalnız raporlama için kullanıldı.
- Ensemble’ın promote edilmesi için en iyi bileşeni validation PR-AUC’de en az `0.001` geçmesi gerekiyordu.

### 13.2 Sonuç

| Validation ağırlık seti | PR-AUC |
|---|---:|
| Random Forest | 0,7250 |
| Engineered LightGBM | 0,7206 |
| Optimize ensemble | **0,7286** |

Seçilen ağırlık:

```text
Random Forest = 0.80
Engineered LightGBM = 0.20
```

Karışım en iyi tek bileşene göre `0.0036` validation PR-AUC artışı sağladı ve promote eşiğini geçti. Final artifact bu nedenle ensemble oldu.

### 13.3 Ensemble test karşılaştırması

| Model | Eşik | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Final ensemble | 0,25 | 0,5870 | **0,7902** | 0,6736 | 0,7448 |
| Engineered LightGBM adayı | 0,33 | **0,6366** | 0,7413 | **0,6850** | **0,7507** |

Ensemble validation protokolüyle promote edildi; testte ise Engineered LightGBM daha yüksek PR-AUC, precision ve F1 verdi. Test selection-only kullanılmadığından final karar değiştirilmedi. Bu sonuç, production kararında validation kazanımı ile model sadeliği/test gözlemi arasında ayrıca yönetişim kararı gerekebileceğini gösterir.

---

## 14. Final model ve test sonuçları

### 14.1 Final artifact

- Model: `Calibrated RF + Engineered LightGBM Ensemble`
- Artifact: `models/best_model.pkl`
- Threshold metadata: `models/threshold.json`
- Ağırlıklar: RF `0.80`, Engineered LightGBM `0.20`
- Feature engineering: LightGBM bileşeninde aktif
- Probability calibration: her bileşende OOF sigmoid/Platt
- Karar eşiği: `0.25`

### 14.2 Final test metrikleri

| Metrik | Değer |
|---|---:|
| ROC-AUC | **0,9320** |
| PR-AUC | **0,7448** |
| Default threshold F1 (`0.50`) | **0,6777** |
| İş threshold F1 (`0.25`) | **0,6736** |
| Precision (`0.25`) | **0,5870** |
| Recall (`0.25`) | **0,7902** |
| Accuracy (`0.25`) | **0,8804** |
| Macro F1 | **0,8002** |
| Weighted F1 | **0,8872** |

Confusion matrix:

|  | Tahmin negatif | Tahmin pozitif |
|---|---:|---:|
| Gerçek negatif | TN = 1.386 | FP = 159 |
| Gerçek pozitif | FN = 60 | TP = 226 |

Yorum:

- Gerçek 286 alıcının 226’sı yakalanmıştır.
- 60 alıcı kaçırılmıştır.
- Satın alma sinyali verilen 385 oturumun 226’sı gerçekten satın almıştır.
- Recall odaklı `0.25` eşiği, default `0.50` eşiğine göre daha fazla fırsat yakalamayı amaçlar.
- Default eşikte F1’in biraz daha yüksek olması, iş maliyeti optimizasyonunun F1 maksimizasyonuyla aynı amaç olmadığını gösterir.

### 14.3 Final model konusunda güven düzeyi

Mevcut veri ve deney protokolü içinde final ensemble seçimi izlenebilir durumdadır çünkü:

- Bileşen ağırlığı testten bağımsız validation yarısında seçilmiştir.
- Validation PR-AUC en iyi bileşenden `0.0036` yüksektir.
- Promote minimumu `0.001` aşılmıştır.
- Calibration iyileşmiştir.
- Test yalnız raporlama için kullanılmıştır.

Ancak tekil Engineered LightGBM testte ensemble’dan daha iyi ranking ve F1 üretmiştir. Modelin “production-ready” olduğu söylenemez; PageValues lineage, gerçek zaman validasyonu ve harici site testi yapılmadan güven düzeyi sınırlıdır.

---

## 15. SHAP açıklanabilirlik

### 15.1 Lokal ve global açıklama farkı

- **Lokal SHAP:** Tek bir oturumun ham model skorunu hangi faktörlerin yukarı/aşağı ittiğini gösterir.
- **Global SHAP:** Referans örneklerde ortalama mutlak SHAP ile modelin hangi özelliklere genel olarak dayandığını gösterir.

Final model ensemble olduğu için mevcut TreeSHAP yalnız Engineered LightGBM bileşenini açıklar; bu bileşenin ensemble ağırlığı `%20`dir. Random Forest’ın `%80` katkısı bu SHAP değerlerinde bulunmaz. SHAP değeri olasılık yüzdesi değildir ve nedensellik ifade etmez.

### 15.2 Global SHAP ilk 15

Hesaplama: temiz train splitinden deterministic 500 satır, `%20` ağırlıklı Engineered LightGBM bileşeni için ortalama mutlak SHAP.

| Sıra | Özellik | Ortalama |SHAP| |
|---:|---|---:|
| 1 | Ziyaret edilen sayfaların dönüşüm değeri | 0,6983 |
| 2 | Yıl içindeki dönemsel ay etkisi (sinüs) | 0,6027 |
| 3 | Ürün sayfası başına dönüşüm değeri | 0,3025 |
| 4 | Dönüşüm değerinin log ölçekli hali | 0,1577 |
| 5 | Ziyaret ayı: Kasım | 0,0962 |
| 6 | Hesap/işlem sayfalarının ziyaret içindeki payı | 0,0581 |
| 7 | Sayfa sonrası siteden çıkış oranı | 0,0492 |
| 8 | Tek sayfadan ayrılma oranı | 0,0474 |
| 9 | Sitede kalma skoru | 0,0436 |
| 10 | Yıl içindeki dönemsel ay etkisi (kosinüs) | 0,0389 |
| 11 | Ürün sayfalarının ziyaret içindeki payı | 0,0382 |
| 12 | Ziyaret ayı: Mayıs | 0,0256 |
| 13 | Hesap ve işlem adımı görüntüleme sayısı | 0,0240 |
| 14 | Sitede geçirilen toplam süre | 0,0236 |
| 15 | Toplam görüntülenen sayfa | 0,0233 |

PageValues ailesinin 1, 3 ve 4. sıralarda olması LightGBM bileşeninin bağımlılığını doğrular. Ensemble’ın tamamı için model-agnostic veya bileşen-bazlı birleşik açıklama ayrıca geliştirilmelidir.

---

## 16. Olasılık segmentasyonu ve iş aksiyonları

İlk sürümde 5 segment kullanıldı. Demo kullanımında ayrımlar fazla ince kaldığı için üç segmente indirildi. Temiz veri final segment sınırları ensemble validation-selected `0.25` eşiği ve yarısı `0.125` ile hizalıdır.

### 16.1 Segment tanımları

| Olasılık | Segment | İş aksiyonu |
|---|---|---|
| `0.000–0.125` | Düşük niyet | Ücretli teşvik verme; düşük maliyetli içerik ve retargeting ile ilgiyi geliştir. |
| `0.125–0.250` | Değerlendirme | Sosyal kanıt, karşılaştırma, stok ve kargo bilgisini görünür kıl. |
| `0.250–1.000` | Yüksek niyet | Checkout desteği ve hatırlatma sun; teşviki kontrollü test ederek marjı koru. |

### 16.2 Test segment sonuçları

| Segment | Oturum | Oturum payı | Gerçek alıcı | Gerçek dönüşüm | Ortalama tahmin | Alıcı yakalama |
|---|---:|---:|---:|---:|---:|---:|
| Düşük | 1.319 | %72,04 | 35 | **%2,65** | %2,63 | %12,24 |
| Değerlendirme | 127 | %6,94 | 25 | **%19,69** | %17,97 | %8,74 |
| Yüksek | 385 | %21,03 | 226 | **%58,70** | %60,53 | **%79,02** |

Gerçek dönüşüm oranının `%2,7 → %19,7 → %58,7` şeklinde monoton yükselmesi segmentlerin anlamlı ayrıştığını gösterir.

İş aksiyonlarının gerçek gelir etkisi model metriklerinden çıkarılamaz; A/B testleriyle ölçülmelidir.

---

## 17. Streamlit demo

### 17.1 Fonksiyonlar

Arayüz dört ana sekmeden oluşur:

1. Manuel tahmin ve what-if.
2. CSV performans testi.
3. Modeli ne etkiliyor / global SHAP.
4. Kullanım rehberi.

Manuel sonuçta şunlar gösterilir:

- Kalibre satın alma olasılığı.
- Üçlü niyet segmenti.
- Binary sinyal.
- Önerilen iş aksiyonu ve hedefi.
- Seçilen sayısal özellik için what-if eğrisi.
- Lokal SHAP pozitif/negatif katkı grafiği.

CSV akışında:

- `Revenue` yoksa toplu probability, segment ve aksiyon üretilir.
- `Revenue` varsa ROC-AUC, PR-AUC, precision, recall, F1, accuracy ve confusion matrix hesaplanır.
- Skorlanmış CSV indirilebilir.

### 17.2 Kullanıcı dostu özellik adları

Teknik feature adları model şeması içinde korunurken arayüzde Türkçe etiketler, yardım balonları ve CSV veri sözlüğü kullanılır. SHAP feature adları da kullanıcı dostu adlara çevrilir.

### 17.3 Sabit analytics demo değerleri

Manuel kullanıcının doğrudan bilemeyeceği analytics alanları sabitlenmiştir:

| Alan | Demo değeri | Gerekçe |
|---|---:|---|
| BounceRates | 0,002899 | Temiz veri medyanı |
| ExitRates | 0,025000 | Temiz veri medyanı |
| PageValues | 0,50 | Kontrollü düşük demo referansı |
| SpecialDay | 0,0 | Nötr gün |

`PageValues=0.50` veri medyanı veya ortalama değildir. Tüm veri medyanı `0`, sıfır-dışı medyan `16,66` olduğu için bunlardan biri kullanıldığında demo sırasıyla düşük veya yüksek skora kilitlenmektedir. `0.50`, davranış alanlarının etkisini gösterecek kontrollü bir demo başlangıcıdır.

Bu sabitleme üretim çözümü değildir. Ayrıca önyüzde gösterilen test PR-AUC `0.7448`, gerçek PageValues dağılımıyla ölçülmüştür; sabit `0.50` PageValues kullanılan manuel demonun ölçülmüş PR-AUC’si değildir.

### 17.4 What-if yorumu

What-if analizi bir özelliği değiştirirken diğerlerini sabit tutar. Model duyarlılığını gösterir; nedensel etki kanıtlamaz. Kullanıcıya açık davranış sayıları/süreleri what-if listesinde tutulmuş, analytics alanları çıkarılmıştır.

---

## 18. PageValues ve data leakage değerlendirmesi

### 18.1 Page Value nedir?

Page Value, analytics sisteminin ziyaret edilen bir sayfanın satın alma veya başka bir dönüşüme ortalama parasal katkısını özetleyen değerdir.

- Ürün fiyatı değildir.
- Satın alma olasılığı değildir.
- Kullanıcının elle bileceği bir alan değildir.
- Gerçek uygulamada analytics altyapısından gelmelidir.

### 18.2 Leakage ne zaman oluşur?

`PageValues` yalnız güçlü olduğu için otomatik olarak leakage sayılmaz. Ancak aşağıdaki durumlardan biri varsa leakage oluşur:

1. Değer mevcut oturumun daha sonraki satın alma sonucuyla hesaplanıyorsa.
2. Tahmin anında henüz mevcut değilse.
3. Validation/test dönemindeki dönüşümler kullanılarak geçmişe dönük hesaplandıysa.
4. Tüm veri döneminde hesaplanan global sayfa değerleri train ve test arasında paylaşılıyorsa.

UCI verisinde gerçek timestamp, sayfa kimliği ve Page Value hesaplama lineage’ı bulunmadığı için bunlar kesin olarak doğrulanamamıştır. Bu nedenle ifade “kanıtlanmış leakage” değil, **yüksek leakage ve erişilebilirlik riski** olmalıdır.

### 18.3 Ablation kanıtı

| Model | CV PR-AUC |
|---|---:|
| PageValues ailesi ile | **0,7562 ± 0,0080** |
| PageValues olmadan | **0,3797 ± 0,0289** |
| Mutlak düşüş | **0,3766** |

Bu büyük düşüş leakage’i tek başına kanıtlamaz; model başarısının büyük kısmının bu bilgi ailesinden geldiğini kanıtlar.

### 18.4 Çözüm seçenekleri

#### En savunulabilir mevcut veri çözümü

Aşağıdaki dört özelliği tamamen kaldırarak modeli yeniden tune etmek:

- `PageValues`
- `LogPageValues`
- `HasPageValue`
- `PageValuePerProductPage`

Bu model daha düşük metrikli fakat erken oturumda erişilebilir bilgilere dayanan bir “production-safe” aday olur.

#### Tarihsel/lagged Page Value

Gerçek timestamp ve page ID mevcutsa Page Value yalnız tahmin tarihinden önce tamamlanmış oturumlardan hesaplanmalıdır. CV sırasında her fold için encoding yalnız fold-train geçmişinden üretilmelidir. Bu, leakage-safe target/statistical encoding yaklaşımıdır.

#### İki model yaklaşımı

- Erken oturum modeli: PageValues yok.
- Geç oturum / analytics modeli: PageValues yalnız tahmin anında gerçekten mevcutsa kullanılır.

---

## 19. Bilinen sınırlamalar

1. **PageValues riski:** Performansın büyük kısmı PageValues ailesine bağlıdır.
2. **Duplicate geçmişi:** 125 exact duplicate artık splitten önce kaldırılmış ve overlap sıfırlanmıştır; bu risk çözülmüştür.
3. **Random split:** Gerçek zaman sırası yoktur; temporal proxy daha düşük sonuç vermiştir.
4. **Tek site:** Veri yalnız tek ve kimliği belirsiz bir e-ticaret sitesine aittir.
5. **Harici validasyon yok:** İkinci site/ülke/dönem testi yapılmamıştır.
6. **Anonim kategoriler:** OS, browser, region ve traffic type kodlarının iş anlamı bilinmemektedir.
7. **Ay eksikleri:** Veri tüm ayları kapsamaz; January ve April yoktur.
8. **Demo sabitleri:** Manuel demo gerçek analytics akışını taklit etmez.
9. **SHAP nedensel değildir:** Model ilişkisini açıklar, müdahale etkisini değil.
10. **Segment aksiyonları test edilmedi:** Ticari fayda A/B testi gerektirir.
11. **Model drift izleme yok:** Production veri ve kalibrasyon kayması henüz ölçülmemektedir.

---

## 20. Önerilen sonraki deneyler

Öncelik sırasıyla:

1. **PageValues-free LightGBM üret.** Dört Page Value özelliğini kaldır, tuning/calibration/threshold/segmentleri yeniden hesapla.
2. **İki modeli yan yana raporla.** Analytics-zengin ve production-safe sonuçlarını ayır.
3. **Ensemble yönetişim kuralını gözden geçir.** Validation kazanımı ile tekil LGB’nin daha iyi test/sadelik sonucunu önceden tanımlı çok ölçütlü kuralla değerlendir.
4. **Ensemble’ın tamamı için açıklama geliştir.** Mevcut SHAP yalnız `%20` LGB bileşenini kapsıyor.
5. **Gerçek timestamp ile forward validation yap.** Rolling window veya expanding window kullan.
6. **Harici site validasyonu yap.** Genellenebilirlik için ikinci kaynak kullan.
7. **Kampanya maliyetlerini iş birimiyle doğrula.** `FN=2, FP=1` varsayımını gerçek TL etkisine bağla.
8. **Segment aksiyonları için A/B test tasarla.** Conversion uplift, incremental revenue ve margin ölç.
9. **Calibration/drift monitoring ekle.** Brier, log loss, PSI/KS ve segment hacimlerini izle.
10. **PageValues lineage dokümante et.** Hangi anda ve hangi tarihsel pencereyle hesaplandığı kesinleştirilsin.

---

## 21. Çalıştırma komutları ve artifact haritası

### 21.1 Ortam

```bash
source .venv/bin/activate
```

Temel paket sürümleri:

- pandas 2.2.3
- numpy 1.26.4
- scikit-learn 1.5.2
- LightGBM 4.5.0
- SHAP 0.46.0
- Streamlit 1.41.1
- Matplotlib 3.8.4

### 21.2 Deney komutları

```bash
# Baseline eğitim ve model karşılaştırması
python -m src.train

# 5-fold tuning, calibration ve robustness
python -m src.tune

# Feature engineering karşılaştırması ve final model
python -m src.feature_experiment

# RF + Engineered LightGBM ensemble
python -m src.ensemble_experiment

# Üç segment analizi
python -m src.segment_analysis

# Global SHAP raporu
python -m src.shap_report

# Headless statik çıktılar
MPLBACKEND=Agg python -m scripts.generate_outputs

# Streamlit
streamlit run app/streamlit_app.py

# Testler
pytest -q
```

### 21.3 Headless grafik politikası

Matplotlib grafikleri:

- `Agg` backend ile çalışır.
- `plt.show()` çağırmaz.
- `outputs/` altında kaydedilir.
- Her figürden sonra `plt.close(fig)` uygulanır.

### 21.4 Model ve rapor artifact’leri

| Dosya | İçerik |
|---|---|
| [`models/best_model.pkl`](models/best_model.pkl) | Final calibrated soft-voting ensemble: `%80` Random Forest + `%20` Engineered LightGBM |
| [`models/threshold.json`](models/threshold.json) | Final ensemble için `0,25` karar eşiği, `%80/%20` ağırlıklar, segment ve test metadata’sı |
| [`reports/model_comparison.json`](reports/model_comparison.json) | Baseline sonuçlar |
| [`reports/tuning_report.json`](reports/tuning_report.json) | Tuning, calibration ve robustness |
| [`reports/feature_engineering_report.json`](reports/feature_engineering_report.json) | Feature deneyleri ve final seçim |
| [`reports/ensemble_report.json`](reports/ensemble_report.json) | Ensemble ağırlık deneyi |
| [`reports/segment_report.json`](reports/segment_report.json) | Üç segment performansı |
| [`reports/global_shap_importance.json`](reports/global_shap_importance.json) | `%20` LGB bileşeni global SHAP sıralaması |

### 21.5 Başlıca görseller

| Görsel | Açıklama |
|---|---|
| [`outputs/target_distribution.png`](outputs/target_distribution.png) | Sınıf dağılımı |
| [`outputs/feature_distributions.png`](outputs/feature_distributions.png) | Sayısal dağılımlar |
| [`outputs/correlation_heatmap.png`](outputs/correlation_heatmap.png) | Korelasyon matrisi |
| [`outputs/roc_curve.png`](outputs/roc_curve.png) | ROC eğrisi |
| [`outputs/pr_curve.png`](outputs/pr_curve.png) | Precision-recall eğrisi |
| [`outputs/confusion_matrix.png`](outputs/confusion_matrix.png) | Final confusion matrix |
| [`outputs/threshold_optimization.png`](outputs/threshold_optimization.png) | Threshold analizi |
| [`outputs/calibration_curve.png`](outputs/calibration_curve.png) | Calibration karşılaştırması |
| [`outputs/stability_analysis.png`](outputs/stability_analysis.png) | Seed stabilitesi |
| [`outputs/shap_summary.png`](outputs/shap_summary.png) | Global SHAP beeswarm |
| [`outputs/global_shap_dashboard.png`](outputs/global_shap_dashboard.png) | Önyüz global SHAP bar grafiği |
| [`outputs/segment_performance.png`](outputs/segment_performance.png) | Segment hacmi ve dönüşümü |
| [`outputs/ensemble_weight_search.png`](outputs/ensemble_weight_search.png) | Ensemble ağırlık taraması |

### 21.6 Test durumu

Son yerel doğrulamada (**5 Ağustos 2026**, Python **3.11.15**) `pytest -q` çıktısı
`28 passed, 14 warnings in 2.73s` oldu. Test kapsamı deduplication, veri doğrulama,
preprocessing davranışı, calibration wrapper, threshold, feature engineering, ensemble,
segment sınırları, batch dashboard, SHAP ranking ve kullanıcı dostu feature adlarını kapsar.

---

## Sonuç

Proje, deduplication, baseline, tuning, calibration, feature engineering, ensemble, SHAP, segmentasyon ve Streamlit demonstrasyonuna kadar tam bir ML yaşam döngüsü sunmaktadır. Duplicate overlap çözülmüş ve tüm artifact’ler 12.205 tekil oturumla yeniden üretilmiştir. Validation protokolü final olarak `%80 RF + %20 Engineered LightGBM` ensemble’ı seçmiştir; testte tekil Engineered LightGBM daha yüksek ranking üretmiştir. En kritik açık konu PageValues bağımlılığıdır; sonraki güvenilirlik adımı PageValues-free yeniden eğitimdir.
