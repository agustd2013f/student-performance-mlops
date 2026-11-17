import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Student_Risk")

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

configs = [
    ("LogisticRegression", {"C": 1.0, "max_iter": 1000}, LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ("RandomForest", {"n_estimators": 150}, RandomForestClassifier(n_estimators=150, random_state=42)),
    ("SVM", {"C": 1.0, "kernel": "rbf"}, SVC(C=1.0, kernel="rbf", probability=True, random_state=42)),
]

best_auc = -1.0
best_run_id = None
best_name = None
best_pipe = None

for name, params, clf in configs:
    with mlflow.start_run(run_name=name) as run:
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        mlflow.log_param("model_type", name)
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(pipe, name="model", serialization_format="pickle")
        print(f"{name:18s} | Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_run_id = run.info.run_id
            best_name = name
            best_pipe = pipe

model_uri = f"runs:/{best_run_id}/model"
mv = mlflow.register_model(model_uri=model_uri, name="StudentRiskClassifier")

client = MlflowClient()
client.set_registered_model_alias(
    name="StudentRiskClassifier",
    alias="champion",
    version=mv.version,
)

joblib.dump(best_pipe, "models/champion.joblib")
print(f"\nModelo registrado: StudentRiskClassifier v{mv.version} ({best_name}) con alias @champion")