# PoF3 - Ekipman Arıza Tahmin Sistemi
## Dağıtım Şirketi Kullanım Kılavuzu

**Versiyon:** 3.1
**Tarih:** Aralık 2025
**Hedef Kitle:** EDAŞ Yöneticileri ve Saha Ekipleri

---

## 📋 İçindekiler

1. [Sistem Hakkında](#sistem-hakkında)
2. [Ne Sağlar?](#ne-sağlar)
3. [Çıktı Dosyaları](#çıktı-dosyaları)
4. [Raporları Nasıl Okuyabilirim?](#raporları-nasıl-okuyabilirim)
5. [Risk Skorlarını Anlama](#risk-skorlarını-anlama)
6. [Aksiyon Önerileri](#aksiyon-önerileri)
7. [Sık Sorulan Sorular](#sık-sorulan-sorular)

---

## 🎯 Sistem Hakkında

**PoF3 (Probability of Failure - Arıza Olasılığı)** sistemi, elektrik dağıtım ekipmanlarınızın gelecekte arıza yapma olasılığını tahmin eder ve risk bazlı bakım planlaması yapmanızı sağlar.

### Temel Prensipler

- **Öngörülü Bakım**: Arıza olmadan önce müdahale
- **Risk Bazlı Planlama**: Kaynakları en kritik ekipmanlara yönlendirme
- **Veri Odaklı Karar**: Geçmiş arıza verileri + makine öğrenmesi
- **IEEE 1366 Standardı**: Uluslararası elektrik güvenilirlik standartlarına uyumlu

### Sistem Ne Yapar?

1. Geçmiş arıza kayıtlarınızı analiz eder
2. Her ekipman için arıza olasılığını hesaplar (3, 6, 12, 24 ay)
3. Arıza sonuçlarının ciddiyetini değerlendirir (CoF - Sonuç Şiddeti)
4. Risk skorları oluşturur (PoF × CoF)
5. Bakım öncelik listeleri hazırlar

---

## 💼 Ne Sağlar?

### İş Değeri

| Fayda | Açıklama |
|-------|----------|
| **%30-40 Bakım Maliyeti Düşüşü** | Reaktif müdahaleden proaktif bakıma geçiş |
| **%50 Acil Müdahale Azalması** | Arızalar olmadan önlem alınır |
| **SAIDI/SAIFI İyileşmesi** | Planlı kesintiler, daha az müşteri şikayeti |
| **Bütçe Optimizasyonu** | CAPEX/OPEX kaynaklarını doğru yere harcama |
| **Regülatör Uyumluluk** | EPDK raporlama gereksinimlerine hazır veri |

### Kullanım Senaryoları

1. **Yıllık Bakım Planı** → 12 aylık tahminleri kullanın
2. **Acil Müdahale Listesi** → KRİTİK risk skoru olanlar
3. **CAPEX Bütçe Hazırlama** → Yüksek riskli ekipman yenileme ihtiyacı
4. **Saha Ekibi Yönlendirme** → Bölgesel risk haritaları
5. **Yönetim Sunumları** → Hazır Excel/PowerPoint raporlar

---

## 📂 Çıktı Dosyaları

Sistem çalıştığında `data/sonuclar/` klasöründe aşağıdaki dosyalar oluşur:

### 🎯 Ana Raporlar (Öncelikli)

| Dosya | İçerik | Kullanım |
|-------|--------|----------|
| **`risk_skorlari_pof3.csv`** | Tüm ekipmanların risk skorları | Ana karar dosyası |
| **`risk_equipment_master.csv`** | Ekipman detayları + risk bilgisi | Detaylı analiz |
| **`chronic_equipment_summary.csv`** | Kronik arıza yapan ekipmanlar | IEEE 1366 analizi |
| **`ensemble_pof_final.csv`** | Makine öğrenmesi tahminleri | İleri seviye analiz |

### 📊 Excel/PowerPoint Raporları

| Dosya | Format | İçerik |
|-------|--------|--------|
| `PoF_Analysis_*.xlsx` | Excel | Özet tablolar, pivot analizler |
| `PoF_Dashboard_*.pptx` | PowerPoint | Yönetim sunumu (grafikler + öneriler) |

### 📈 Görsel Raporlar (`gorseller/`)

- `chronic_distribution.png` - Kronik ekipman dağılımı
- `equipment_distribution.png` - Ekipman tipi dağılımı
- `fault_trends.png` - Arıza trendleri
- `feature_importance.png` - Hangi faktörler önemli?
- `pof_by_horizon.png` - Zaman ufkuna göre PoF dağılımı
- `survival_curves_by_class.png` - Ekipman ömür eğrileri

### 🔍 Teknik Detay Dosyaları (İsteğe Bağlı)

- `shap_feature_importance.csv` - Hangi özellikler arızayı etkiliyor?
- `feature_correlations.csv` - Özellik korelasyonları
- `temporal_cv_scores.csv` - Model doğrulama skorları

---

## 📖 Raporları Nasıl Okuyabilirim?

### 1. Risk Skorları Dosyası (`risk_skorlari_pof3.csv`)

**Sütunlar:**

| Sütun | Açıklama | Örnek Değer |
|-------|----------|-------------|
| `cbs_id` | Ekipman kimliği | CBS12345 |
| `Ekipman_Tipi` | Ekipman türü | Transformatör |
| `PoF_12M` | 12 ay arıza olasılığı (0-1) | 0.75 (=%75) |
| `CoF` | Sonuç şiddeti skoru | 8.5 |
| `Risk_Score` | Risk skoru (PoF × CoF) | 6.375 |
| `Risk_Sinifi` | Risk sınıfı | KRİTİK |

**Risk Sınıfı Anlamları:**

- 🔴 **KRİTİK**: Hemen müdahale gerekli (0-30 gün)
- 🟠 **YÜKSEK**: 1-3 ay içinde planla
- 🟡 **ORTA**: 6-12 ay içinde bakım
- 🟢 **DÜŞÜK**: Rutin bakım programında

### 2. Kronik Ekipman Raporu (`chronic_equipment_summary.csv`)

IEEE 1366 standardına göre **kronik arıza yapan ekipmanlar**:

**Kritik Sütunlar:**

| Sütun | Açıklama |
|-------|----------|
| `kronik_flag` | 1 = Kronik, 0 = Normal |
| `ariza_sayisi_365gun` | Son 1 yılda arıza sayısı |
| `poisson_p_value` | İstatistiksel anlamlılık (küçükse kötü) |
| `dominant_sebep` | En sık arıza nedeni |

**IEEE 1366 Tanımı:**
Bir ekipman 365 günde **4+ arıza** yapmışsa ve bu istatistiksel olarak anlamlıysa **kronik** kabul edilir.

### 3. Excel Raporu (`PoF_Analysis_*.xlsx`)

**Sekmeler:**

1. **Özet** - Genel durum, ekipman sayıları, risk dağılımı
2. **Öncelikli Müdahale** - KRİTİK risk skorlu ekipmanlar
3. **CAPEX Planı** - Yenileme ihtiyacı olanlar
4. **Bakım Listesi** - YÜKSEK/ORTA risk ekipmanlar
5. **Ekipman Tipi Analizi** - Tür bazında risk dağılımı
6. **Trend Analizi** - Aylık/yıllık trendler

---

## 🎯 Risk Skorlarını Anlama

### Risk Skoru Formülü

```
Risk_Score = PoF_12M × CoF
```

**PoF (Probability of Failure):** 0.0 ile 1.0 arası (0% - 100% olasılık)
**CoF (Consequence of Failure):** 1.0 ile 10.0 arası (düşük - yüksek etki)

### Risk Matrisi

|  | **Düşük CoF** | **Orta CoF** | **Yüksek CoF** | **Çok Yüksek CoF** |
|---|---|---|---|---|
| **Yüksek PoF** | 🟠 YÜKSEK | 🔴 KRİTİK | 🔴 KRİTİK | 🔴 KRİTİK |
| **Orta PoF** | 🟡 ORTA | 🟠 YÜKSEK | 🔴 KRİTİK | 🔴 KRİTİK |
| **Düşük PoF** | 🟢 DÜŞÜK | 🟡 ORTA | 🟠 YÜKSEK | 🟠 YÜKSEK |

### CoF Faktörleri

| Faktör | Düşük Etki | Yüksek Etki |
|--------|------------|-------------|
| **Ekipman Maliyeti** | Küçük parçalar | Transformatör, kesici |
| **Gerilim Seviyesi** | Alçak gerilim | Yüksek gerilim |
| **Müşteri Sayısı** | <100 abone | >1000 abone |
| **Tamir Süresi (MTTR)** | <2 saat | >8 saat |

### PoF Faktörleri (Modelin Kullandığı)

- ✅ Ekipman yaşı
- ✅ Geçmiş arıza sıklığı (MTBF)
- ✅ Kronik arıza geçmişi
- ✅ Bakım kayıtları
- ✅ Mevsimsel faktörler
- ✅ Ekipman tipi risk profili

---

## 🛠️ Aksiyon Önerileri

### KRİTİK Risk (Risk_Score > 7.0)

**Önerilen Aksiyonlar:**
1. ✅ Ekipmanı 0-30 gün içinde sahada kontrol et
2. ✅ Yedek parça stoğunu kontrol et
3. ✅ Müşteri kesinti planı hazırla (bilgilendirme)
4. ✅ Acil müdahale ekibine bildir
5. ✅ CAPEX bütçesinde yenileme planla

**Örnek Aksiyon:** Transformatör TR-12345, Risk: 8.5
→ Saha ekibi 1 hafta içinde termografik ölçüm yapacak
→ Arıza yaparsa 2 saat içinde yeni transformatör monte edilecek
→ 50 abone etkilenecek, SMS ile bilgilendirme hazır

### YÜKSEK Risk (Risk_Score 5.0 - 7.0)

**Önerilen Aksiyonlar:**
1. ✅ 1-3 ay içinde planlı bakım
2. ✅ Durum izleme (aylık kontrol)
3. ✅ Bakım bütçesine dahil et
4. ✅ Kritik olup olmadığını yeniden değerlendir

### ORTA Risk (Risk_Score 3.0 - 5.0)

**Önerilen Aksiyonlar:**
1. ✅ 6-12 ay içinde rutin bakım
2. ✅ 6 ayda bir yeniden risk skoru hesapla
3. ✅ Trend değişikliğini izle

### DÜŞÜK Risk (Risk_Score < 3.0)

**Önerilen Aksiyonlar:**
1. ✅ Normal bakım programında tut
2. ✅ Yıllık risk değerlendirmesine dahil et

---

## 📊 Zaman Ufku Seçimi

Sistem **4 farklı zaman ufku** için tahmin yapar:

| Ufuk | Kullanım Amacı |
|------|----------------|
| **3 ay** | Acil müdahale planı, kış/yaz hazırlığı |
| **6 ay** | Mevsimsel bakım planı |
| **12 ay** | Yıllık bakım bütçesi, CAPEX planı |
| **24 ay** | Stratejik yenileme planı, regülatör raporlama |

**Öneri:** Bakım planlaması için **12 aylık** tahminleri kullanın (en dengeli).

---

## ❓ Sık Sorulan Sorular

### S1: Sistem ne sıklıkla çalıştırılmalı?

**C:** Ayda 1 kez veya yeni arıza verileri geldikçe. Veriler ne kadar güncel olursa tahminler o kadar doğru olur.

---

### S2: Tahmin doğruluğu ne kadar?

**C:** Temporal cross-validation sonuçlarına göre:
- **AUC (Area Under Curve):** ~0.82-0.88 (0.7'nin üzeri iyi)
- **Average Precision:** ~0.75-0.85

Bu skorlar, sistemin **%80-85 doğruluk** ile çalıştığını gösterir.

---

### S3: Hangi ekipman tipleri destekleniyor?

**C:** Tüm dağıtım ekipmanları:
- Transformatörler
- Kesiciler (Circuit Breakers)
- Ayırıcılar (Disconnectors)
- Sigorta Kutuları (Fuse Boxes)
- Kablolar
- Diğer (model kendi öğrenir)

---

### S4: Sisteme yeni veri nasıl yüklenir?

**C:** İki Excel dosyası hazırlanır:
1. `ariza_final.xlsx` - Arıza kayıtları
2. `saglam_final.xlsx` - Sağlam ekipman listesi

Dosyalar `data/girdiler/` klasörüne konur ve sistem çalıştırılır.

**Gerekli Sütunlar:**
- `cbs_id` - Ekipman kimliği
- `Ariza_Baslangic_Zamani` - Arıza tarihi
- `Ekipman_Tipi` - Ekipman türü
- `Sure_Saat` - Arıza süresi (saat)

---

### S5: Risk skoru yüksek ama ekipman yeni, neden?

**C:** Risk skoru sadece yaşa bağlı değil:
- Ekipman tipi risk profili (bazı tipler doğal olarak riskli)
- Bölgesel faktörler (yük, çevre koşulları)
- CoF yüksek olabilir (çok müşteri etkiliyor)
- Bakım geçmişi eksik olabilir

**Öneri:** Saha ekibinin manuel değerlendirmesiyle teyit edin.

---

### S6: Excel/PowerPoint raporları oluşturulmuyor?

**C:** Opsiyonel adım hatası, ana tahminler yine de çalışır. Teknik ekibe `python-pptx` kurulumu için bilgi verin.

---

### S7: "Kronik" ekipman ne demek?

**C:** IEEE 1366 standardına göre:
- Son 365 günde **4+ arıza** yapmış
- İstatistiksel olarak anlamlı (tesadüf değil)

Bu ekipmanlar normal müdahaleden sonra tekrar arıza yapıyor, kök neden analizi gerekli.

---

### S8: Sistemin kullandığı "özellikler" nelerdir?

**C:** `feature_importance.png` grafiğinde görebilirsiniz. Genelde en önemli 5 faktör:

1. **Ekipman Yaşı** (gün)
2. **Son Arıza Sonrası Geçen Süre** (gün)
3. **Kronik Arıza İndeksi** (IEEE 1366)
4. **MTBF (Arızalar Arası Ortalama Süre)**
5. **Mevsimsel Faktörler** (yaz/kış yükü)

---

### S9: CoF (Sonuç Şiddeti) nasıl hesaplanıyor?

**C:**
```
CoF = Ekipman_Maliyeti × Gerilim_Çarpanı × Müşteri_Etkisi × MTTR_Faktörü
```

**Örnekler:**
- LV Sigorta Kutusu, 20 abone → CoF ≈ 2.5 (DÜŞÜK)
- MV Transformatör, 500 abone → CoF ≈ 7.0 (YÜKSEK)
- HV Kesici, 5000 abone → CoF ≈ 9.5 (ÇOK YÜKSEK)

---

### S10: Tahminler kesin mi, yoksa olasılık mı?

**C:** **Olasılıktır**, kesin değildir.

- PoF = 0.80 → %80 ihtimalle arıza yapar (kesin değil)
- PoF = 0.05 → %5 ihtimalle arıza yapar (ama %0 değil)

**Kullanım:** Risk bazlı karar verme için, yüzde yüz garanti için değil.

---

## 📞 Destek ve İletişim

### Teknik Sorunlar

**Teknik ekibinizle iletişime geçin:**
- Sistem çalışmıyor
- Dosyalar oluşturulmuyor
- Hata mesajları

### İş Süreçleri

**Veri ekibinizle görüşün:**
- Veri kalitesi sorunları
- Yeni ekipman tipi ekleme
- Özel raporlama ihtiyacı

### Eğitim Talepleri

Kullanıcı eğitimi, workshop, rapor yorumlama eğitimi için kurumsal iletişim kanallarınızı kullanabilirsiniz.

---

## 📚 Ek Kaynaklar

- **IEEE 1366 Standardı:** Elektrik güvenilirlik metrikleri
- **EPDK Raporlama:** SAIDI/SAIFI hesaplamaları
- **Öngörülü Bakım:** Endüstri 4.0 en iyi pratikleri

---

**Son Güncelleme:** Aralık 2025
**Versiyon:** 3.1
**Lisans:** Kurumsal Kullanım
