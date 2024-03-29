# app/Dockerfile

FROM python:3.7-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

ARG PORT=8501

HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health

ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port", "${PORT}", "--server.address=0.0.0.0"]