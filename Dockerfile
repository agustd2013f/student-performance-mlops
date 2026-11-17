FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pandas==2.2.3 \
    scikit-learn==1.9.0 \
    joblib \
    jinja2 \
    python-multipart

COPY app ./app
COPY templates ./templates
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]