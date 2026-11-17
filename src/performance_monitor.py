import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

model = joblib.load("models/champion.joblib")

ref = pd.read_csv("data/reference.csv")
cur = pd.read_csv("data/current.csv")

X_ref, y_ref = ref.drop(columns=["riesgo_reprobacion"]), ref["riesgo_reprobacion"]
X_cur, y_cur = cur.drop(columns=["riesgo_reprobacion"]), cur["riesgo_reprobacion"]

pred_ref, prob_ref = model.predict(X_ref), model.predict_proba(X_ref)[:, 1]
pred_cur, prob_cur = model.predict(X_cur), model.predict_proba(X_cur)[:, 1]

acc_ref, f1_ref, auc_ref = accuracy_score(y_ref, pred_ref), f1_score(y_ref, pred_ref), roc_auc_score(y_ref, prob_ref)
acc_cur, f1_cur, auc_cur = accuracy_score(y_cur, pred_cur), f1_score(y_cur, pred_cur), roc_auc_score(y_cur, prob_cur)

print("=== MONITOREO DE RENDIMIENTO DEL MODELO ===")
print(f"Referencia (Math)       -> Accuracy: {acc_ref:.4f} | F1: {f1_ref:.4f} | ROC-AUC: {auc_ref:.4f}")
print(f"Actual     (Portuguese) -> Accuracy: {acc_cur:.4f} | F1: {f1_cur:.4f} | ROC-AUC: {auc_cur:.4f}")

caida_f1 = f1_ref - f1_cur
caida_auc = auc_ref - auc_cur
print(f"\nDiferencia F1: {caida_f1:+.4f} | Diferencia ROC-AUC: {caida_auc:+.4f}")

if caida_f1 > 0.10 or caida_auc > 0.10:
    print("ALERTA: revisar rendimiento (degradación superior al umbral de 0.10)")
else:
    print("Rendimiento dentro del umbral permitido")