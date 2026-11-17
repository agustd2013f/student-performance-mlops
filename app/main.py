from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.schemas import StudentInput

app = FastAPI(title="Student Risk Prediction API", version="1.0.0")
templates = Jinja2Templates(directory="templates")

MODEL_PATH = Path("models/champion.joblib")
model = joblib.load(MODEL_PATH)


def classify_risk(prob: float) -> str:
    if prob < 0.35:
        return "Bajo"
    if prob < 0.65:
        return "Medio"
    return "Alto"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
def predict(student: StudentInput):
    df = pd.DataFrame([student.model_dump()])
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return {
        "riesgo_predicho": pred,
        "probabilidad_riesgo": round(prob, 4),
        "risk_level": classify_risk(prob),
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"result": None})


@app.post("/", response_class=HTMLResponse)
def home_predict(
    request: Request,
    school: str = Form(...),
    sex: str = Form(...),
    age: int = Form(...),
    studytime: int = Form(...),
    failures: int = Form(...),
    famsup: str = Form(...),
    higher: str = Form(...),
    internet: str = Form(...),
    goout: int = Form(...),
    absences: int = Form(...),
):
    payload = StudentInput(
        school=school,
        sex=sex,
        age=age,
        studytime=studytime,
        failures=failures,
        famsup=famsup,
        higher=higher,
        internet=internet,
        goout=goout,
        absences=absences,
    )
    result = predict(payload)
    return templates.TemplateResponse(request=request, name="index.html", context={"result": result})