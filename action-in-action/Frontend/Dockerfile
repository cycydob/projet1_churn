FROM python:3.11.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY frontend.py .

RUN mkdir -p .streamlit && \
    printf '[server]\nheadless = true\nport = 8501\naddress = "0.0.0.0"\nenableCORS = false\n\n[browser]\ngatherUsageStats = false\n' > .streamlit/config.toml

EXPOSE 8501
ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]