FROM python:3.11.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY churn_model.pkl encoders.pkl scaler.pkl feature_names.pkl backend.py ./

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]