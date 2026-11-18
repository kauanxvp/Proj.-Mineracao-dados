FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Treina durante o build (a base está em data/raw): a imagem já sai com modelo, banco e runs do MLflow
RUN rm -rf mlflow.db mlruns && python -m src.pipeline && python -m pytest -q
EXPOSE 8000 8501 5000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
