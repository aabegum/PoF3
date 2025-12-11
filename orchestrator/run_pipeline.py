import subprocess
import sys
import os
import time
import logging
from datetime import datetime

# UTF-8 console encoding
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Proje Kökü
PROJE_KOKU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------------------
# PIPELINE DEFINITION (Updated Structure)
# Format: (Step Name, Script Path, Is Critical?)
# ------------------------------------------------------------------------------
ADIMLAR = [
    # 1. Data Cleaning & Integration
    ("01_veri_isleme",          os.path.join("pipeline", "01_veri_isleme.py"), True),
    
    # 2. Feature Engineering (Physics + History)
    ("02_ozellik_muhendisligi", os.path.join("pipeline", "02_ozellik_muhendisligi.py"), True),
    
    # 3. AI Models (Cox + Weibull + RSF + ML + Ensemble)
    # Note: Ensure your script is named '03_hibrit_model.py' or update here
    ("03_hibrit_model",         os.path.join("pipeline", "03_hibrit_model.py"), True),
    
    # 4. Chronic Analysis (IEEE 1366)
    ("04_tekrarlayan_ariza",    os.path.join("pipeline", "04_tekrarlayan_ariza.py"), True),
    
    # 4b. Risk Scoring (CoF * PoF) - NEW STEP
    ("04_risk_scoring",         os.path.join("pipeline", "04_risk_scoring.py"), True),
    
    # 5. Reporting & Visualization (Merged 05+06+07) - NEW STEP
    ("05_raporlama",            os.path.join("pipeline", "05_raporlama_ve_gorsellestirme.py"), False),
]

def setup_master_logger():
    log_dir = os.path.join(PROJE_KOKU, "loglar")
    os.makedirs(log_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"pipeline_master_{ts}.log")

    logger = logging.getLogger("pipeline_master")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger, log_path

def calistir_adim(adim_adi: str, script_yolu: str, is_critical: bool, logger: logging.Logger) -> tuple:
    tam_yol = os.path.join(PROJE_KOKU, script_yolu)

    if not os.path.exists(tam_yol):
        msg = f"[HATA] Script bulunamadi: {tam_yol}"
        if is_critical:
            logger.error(msg)
            raise FileNotFoundError(msg)
        else:
            logger.warning(f"[WARN] {msg} - Atlaniyor.")
            return 0.0, False

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"[PIPELINE] BASLATIYOR: {adim_adi}")
    logger.info(f"   -> {script_yolu}")
    logger.info("=" * 80)

    start = time.time()
    cmd = [sys.executable, tam_yol]
    
    # Run script
    sonuc = subprocess.run(cmd, cwd=PROJE_KOKU, shell=False)
    sure = time.time() - start

    if sonuc.returncode != 0:
        msg = f"[HATA] {adim_adi} basarisiz oldu. Kod: {sonuc.returncode}"
        if is_critical:
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            logger.warning(f"[WARN] {msg} - Devam ediliyor.")
            return sure, False

    logger.info(f"[OK] {adim_adi} tamamlandi ({sure:.1f} sn).")
    return sure, True

def main():
    logger, log_path = setup_master_logger()

    logger.info("=" * 80)
    logger.info("PoF3 PIPELINE ORCHESTRATOR (v4.0 Final)")
    logger.info(f"Zaman: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    total_time = 0.0
    results = []

    try:
        for name, path, crit in ADIMLAR:
            duration, success = calistir_adim(name, path, crit, logger)
            results.append((name, duration, success))
            total_time += duration

        logger.info("")
        logger.info("=" * 80)
        logger.info("PIPELINE OZET RAPORU")
        logger.info("=" * 80)
        
        for name, dur, success in results:
            status = "BASARILI" if success else "BASARISIZ"
            logger.info(f"  {status:10s} {name:30s} : {dur:6.1f} sn")
            
        logger.info("-" * 80)
        logger.info(f"Toplam Sure: {total_time:.1f} sn")
        logger.info(f"Log Dosyasi: {log_path}")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception("PIPELINE KRITIK HATA ILE DURDU")
        sys.exit(1)

if __name__ == "__main__":
    main()