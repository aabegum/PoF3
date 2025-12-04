from pipeline import (
    01_data_processing as dp,
    02_feature_engineering as fe,
    03_survival_models as sm,
    04_chronic_detection as ch,
    05_risk_assessment as ra
)

def run_all():
    dp.run()
    fe.run()
    sm.run()
    ch.run()
    ra.run()

if __name__ == '__main__':
    run_all()
