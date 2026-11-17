from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

df = pd.read_csv("data/processed/student_clean.csv")
X = df.drop(columns=["riesgo_reprobacion"])
y = df["riesgo_reprobacion"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_cols = ["age", "studytime", "failures", "goout", "absences"]
cat_cols = ["school", "sex", "famsup", "higher", "internet"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
}

best_name = None
best_auc = -1.0
best_pipeline = None

print("=== RESULTADOS DE ENTRENAMIENTO ===")
for name, clf in models.items():
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", clf),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"{name:18s} | Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_name = name
        best_pipeline = pipe

Path("models").mkdir(exist_ok=True)
joblib.dump(best_pipeline, "models/champion.joblib")
print(f"\nModelo ganador guardado: {best_name} (ROC-AUC = {best_auc:.4f}) -> models/champion.joblib")