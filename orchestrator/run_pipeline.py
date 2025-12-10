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

# Orchestrator klasöründen bir üst klasörü proje kökü yap
PROJE_KOKU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Core pipeline steps (implemented and tested)
ADIMLAR = [
    ("01_veri_isleme",          os.path.join("pipeline", "01_veri_isleme.py"), True),   # Critical
    ("02_ozellik_muhendisligi", os.path.join("pipeline", "02_ozellik_muhendisligi.py"), True),   # Critical
    ("03_sagkalim_modelleri",   os.path.join("pipeline", "03_sagkalim_modelleri.py"), True),   # Critical
    ("04_tekrarlayan_ariza",    os.path.join("pipeline", "04_tekrarlayan_ariza.py"), True),   # Critical
    ("05_risk_degerlendirme",   os.path.join("pipeline", "05_risk_degerlendirme.py"), False),  # Optional (needs CoF)
    ("06_gorsellestirmeler",    os.path.join("pipeline", "06_gorsellestirmeler.py"), False),  # Optional
]


def setup_master_logger():
    """Setup master pipeline logger that writes to file and console."""
    log_dir = os.path.join(PROJE_KOKU, "loglar")
    os.makedirs(log_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"pipeline_master_{ts}.log")

    logger = logging.getLogger("pipeline_master")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger, log_path


def calistir_adim(adim_adi: str, script_yolu: str, is_critical: bool, logger: logging.Logger) -> tuple:
    """
    Tek bir pipeline adımını çalıştırır.

    Args:
        adim_adi: Step name
        script_yolu: Script path
        is_critical: If True, failure stops pipeline. If False, failure is logged but pipeline continues.
        logger: Master logger

    Returns:
        (duration, success): Tuple of duration in seconds and success boolean
    """
    tam_yol = os.path.join(PROJE_KOKU, script_yolu)

    if not os.path.exists(tam_yol):
        msg = f"[HATA] Adim script'i bulunamadi: {tam_yol}"
        if is_critical:
            logger.error(msg)
            raise FileNotFoundError(msg)
        else:
            logger.warning(f"[WARN] {msg} - Atlaniyor (opsiyonel adim)")
            return 0.0, False

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"[PIPELINE] {adim_adi} baslatiliyor -> {script_yolu}")
    logger.info("=" * 80)

    start = time.time()

    # Aynı Python yorumlayıcısını kullan (venv güvenli)
    cmd = [sys.executable, tam_yol]

    # Çıktıyı direkt konsola akıtıyoruz; log dosyaları step içinde zaten tutuluyor
    sonuc = subprocess.run(
        cmd,
        cwd=PROJE_KOKU,
        shell=False
    )

    sure = time.time() - start

    if sonuc.returncode != 0:
        msg = f"[HATA] {adim_adi} adimi hata ile sonlandi. Return code: {sonuc.returncode}"
        if is_critical:
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            logger.warning(f"[WARN] {msg} - Devam ediliyor (opsiyonel adim)")
            return sure, False

    logger.info(f"[PIPELINE] {adim_adi} basariyla tamamlandi. Sure: {sure: .1f} sn")
    return sure, True


def main():
    logger, log_path = setup_master_logger()

    logger.info("")
    logger.info("=" * 80)
    logger.info("PoF3 PIPELINE ORCHESTRATOR")
    logger.info("=" * 80)
    logger.info(f"Calistirma zamani: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Proje koku       : {PROJE_KOKU}")
    logger.info(f"Python yorumlayici: {sys.executable}")
    logger.info(f"Master log dosyasi: {log_path}")
    logger.info("=" * 80)
    logger.info("")

    toplam_sure = 0.0
    adim_sureleri = []
    basarili_adimlar = []
    basarisiz_adimlar = []

    try:
        for adim_adi, script_yolu, is_critical in ADIMLAR:
            sure, success = calistir_adim(adim_adi, script_yolu, is_critical, logger)
            adim_sureleri.append((adim_adi, sure, success))
            toplam_sure += sure

            if success:
                basarili_adimlar.append(adim_adi)
            else:
                basarisiz_adimlar.append(adim_adi)

        logger.info("")
        logger.info("=" * 80)
        if basarisiz_adimlar:
            logger.info("[PIPELINE] KRITIK ADIMLAR TAMAMLANDI (BAZI OPSIYONEL ADIMLAR ATLANABILIR)")
        else:
            logger.info("[PIPELINE] TUM ADIMLAR BASARIYLA TAMAMLANDI")
        logger.info("=" * 80)
        logger.info("Tamamlanan adimlar:")
        for adim_adi, s, success in adim_sureleri:
            status = "[OK]" if success else "[SKIP]"
            logger.info(f"  {status} {adim_adi:30s}: {s:6.1f} sn")
        logger.info("-" * 80)
        logger.info(f"Toplam sure: {toplam_sure: .1f} sn")
        logger.info(f"Basarili: {len(basarili_adimlar)}/{len(ADIMLAR)}")
        if basarisiz_adimlar:
            logger.info(f"Atlanan opsiyonel adimlar: {', '.join(basarisiz_adimlar)}")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Olusturulan ciktilar:")
        logger.info("  - Cox PoF tahminleri (3, 6, 12, 24 ay)")
        logger.info("  - RSF PoF tahminleri (3, 6, 12, 24 ay) + Feature Importance")
        logger.info("  - ML PoF tahminleri (leakage-free, 2 reference windows)")
        logger.info("  - Temporal CV robustness scores")
        logger.info("  - SHAP feature importance")
        logger.info("  - Kronik ekipman analizi (cok seviyeli)")
        logger.info("  - Survival curves (gorseller/)")
        logger.info("  - Turkce dokumantasyon (OKUBBENI.txt)")
        logger.info("")
        logger.info("Cikti dizini: data/sonuclar/")
        logger.info(f"Master log: {log_path}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error("[PIPELINE] HATA OLUSTU - PIPELINE DURDURULDU")
        logger.error("=" * 80)
        logger.error(str(e))
        logger.error("")
        logger.error("Detayli hata loglari ilgili adimin kendi log dosyasinda (loglar/ veya LOG_DIR) yer aliyor.")
        logger.error(f"Master pipeline log: {log_path}")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
