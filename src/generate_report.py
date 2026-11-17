from pathlib import Path
import pandas as pd

ref = pd.read_csv("data/reference.csv").drop(columns=["riesgo_reprobacion"])
cur = pd.read_csv("data/current.csv").drop(columns=["riesgo_reprobacion"])

Path("reports").mkdir(exist_ok=True)
out_html = "reports/data_drift.html"

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=ref, current_data=cur)
    snapshot.save_html(out_html)
except Exception:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    report.save_html(out_html)

print(f"Reporte de Data Drift generado en: {out_html}")