import pandas as pd

cols = [
    "school", "sex", "age", "studytime", "failures",
    "famsup", "higher", "internet", "goout", "absences",
    "riesgo_reprobacion",
]

ref = pd.read_csv("data/raw/student-mat.csv", sep=";")
ref["riesgo_reprobacion"] = (ref["G3"] < 10).astype(int)
ref = ref[cols].dropna()

cur = pd.read_csv("data/raw/student-por.csv", sep=";")
cur["riesgo_reprobacion"] = (cur["G3"] < 10).astype(int)
cur = cur[cols].dropna()

ref.to_csv("data/reference.csv", index=False)
cur.to_csv("data/current.csv", index=False)

print("Shapes -> Referencia (Math):", ref.shape, "| Actual (Portuguese):", cur.shape)
print("\n--- Resumen Referencia (student-mat) ---")
print(ref[["age", "studytime", "failures", "absences"]].describe().round(2))
print("\n--- Resumen Actual (student-por) ---")
print(cur[["age", "studytime", "failures", "absences"]].describe().round(2))