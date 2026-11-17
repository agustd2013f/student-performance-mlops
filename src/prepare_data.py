from pathlib import Path
import pandas as pd

raw_path = Path("data/raw/student-mat.csv")
df = pd.read_csv(raw_path, sep=";")

# Variable objetivo: 1 = riesgo de reprobar (G3 < 10), 0 = aprueba
df["riesgo_reprobacion"] = (df["G3"] < 10).astype(int)

selected_cols = [
    "school", "sex", "age", "studytime", "failures",
    "famsup", "higher", "internet", "goout", "absences",
    "riesgo_reprobacion",
]

clean_df = df[selected_cols].dropna().copy()

out_path = Path("data/processed/student_clean.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
clean_df.to_csv(out_path, index=False)

print(f"Dataset limpio guardado en {out_path} -> Filas: {len(clean_df)}, Columnas: {clean_df.shape[1]}")
print("Distribución de riesgo_reprobacion:\n", clean_df["riesgo_reprobacion"].value_counts())