# PoF3 – Varlık Yönetimi – Arıza İlişkisi (Türk EDAŞ)

> **Amaç:** Elektrik dağıtım ekipmanları için arıza olasılığı (PoF), kronik arızalar ve risk skorlarının
> **standartlara uyumlu**, **tekrarlanabilir** ve **Türkçe çıktılarla** hesaplanması.

---

## 1. Pipeline Genel Bakış

PoF3 dört ana analitik adım + opsiyonel görselleştirme modülünden oluşur:

1. **01_data_processing.py**  
   - Arıza + sağlıklı ekipman verilerini okur  
   - Kullanılacak sütunları uygular (cause code, started at, ended at, duration time, cbs_id, Sebekeye_Baglanma_Tarihi, vb.)  
   - Mixed dataset (sağlıklı + arızalı) yapısını hazırlar  
   - Çıktılar:
     - `data/intermediate/fault_events_clean.csv`
     - `data/intermediate/equipment_master.csv`
     - `data/intermediate/survival_base.csv`

2. **02_feature_engineering.py**  
   - Ekipman bazlı özellikleri üretir:
     - Toplam arıza sayısı
     - MTBF (gün)
     - Kronik 90 gün flag (`Chronic_90d_Flag` / `Kronik_90G_Flag`)
   - Çıktı:
     - `data/intermediate/features_pof3.csv`
test set cross validation
3. **03_survival_models.py**  
   - `survival_base` + `features_pof3` üzerinden Cox PH ile survival modeli kurar  
   - 3M / 6M / 12M /24M (ekle) için PoF (arıza olasılığı) üretir  
   - Çıktılar:
     - `data/outputs/pof_cox_3m.csv`
     - `data/outputs/pof_cox_6m.csv`(gerek yok gibi)
     - `data/outputs/pof_cox_12m.csv`

4. **04_chronic_detection.py**  
   - IEEE benzeri kronik tanımı uygular:
     - Son 12 ayda ≥3 arıza
     - Yıllık arıza oranı (λ)  
   - Mevcut 90 günlük kronik flag ile birleşik kronik flag üretir  
   - Çıktılar:
     - `data/outputs/chronic_equipment_summary.csv`
     - `data/outputs/chronic_equipment_only.csv`

5. **05_risk_assessment.py**  
   - 12 aylık PoF + CoF skorlarını birleştirir  
   - `Risk_Skoru = PoF_12M * CoF_Skoru` hesaplar  
   - `Risk_Sinifi` (DÜŞÜK / ORTA / YÜKSEK) üretir  
   - Çıktılar:
     - `data/outputs/pof3_risk_table.csv`
     - `data/outputs/pof3_risk_summary_by_type.csv`

6. **06_visualizations.py** (opsiyonel)  
   - Açıklayıcı grafikler üretir:
     - Ekipman yaşı dağılımı
     - Ekipman tipi dağılımı
     - Kaplan-Meier survival eğrileri
     - Kronik ekipman ısı haritaları
     - Risk matrisi ve risk dağılımları  
   - Çıktılar:
     - `visuals/plots_survival/*.png`
     - `visuals/plots_chronic/*.png`
     - `visuals/plots_risk/*.png`

---

## 2. Klasör Yapısı

```text
PoF3/
│
├── config/
│   └── config.py
│
├── data/
│   ├── inputs/
│   │   ├── fault_merged_data.xlsx      # Arıza kayıtları
│   │   ├── health_merged_data.xlsx     # Sağlıklı ekipman listesi
│   │   └── cof_data.(csv|xlsx)         # CoF skorları (isteğe bağlı)
│   ├── intermediate/
│   └── outputs/
│
├── docs/
│   ├── DATA_CONTRACT.md
│   ├── PIPELINE_OVERVIEW.md
│   ├── METHODS.md
│   └── CHANGELOG.md
│
├── pipeline/
│   ├── 01_data_processing.py
│   ├── 02_feature_engineering.py
│   ├── 03_survival_models.py
│   ├── 04_chronic_detection.py
│   ├── 05_risk_assessment.py
│   └── 06_visualizations.py
│
├── orchestrator/
│   └── run_pipeline.py                 # (opsiyonel) tek komutla tüm adımlar
│
├── utils/
│   └── logger.py
│
├── visuals/
│   ├── plots_survival/
│   ├── plots_chronic/
│   └── plots_risk/
│
└── README.md
