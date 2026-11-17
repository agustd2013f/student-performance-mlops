import joblib
import pandas as pd

model = joblib.load("models/champion.joblib")

ref = pd.read_csv("data/reference.csv").drop(columns=["riesgo_reprobacion"])
cur = pd.read_csv("data/current.csv").drop(columns=["riesgo_reprobacion"])

ref_probs = pd.Series(model.predict_proba(ref)[:, 1], name="prob_riesgo_ref")
cur_probs = pd.Series(model.predict_proba(cur)[:, 1], name="prob_riesgo_cur")

print("=== MONITOREO DE PREDICCIONES (PROBABILIDAD DE RIESGO) ===")
print("\nCohorte Referencia (Matemáticas):")
print(ref_probs.describe().round(4))
print("\nCohorte Actual (Portugués):")
print(cur_probs.describe().round(4))