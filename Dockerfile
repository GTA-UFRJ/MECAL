FROM python:3.11-slim

WORKDIR /app

# Dependências de sistema mínimas para sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código-fonte (datasets são montados em runtime)
COPY preprocessing.py .
COPY MECAL.py .
COPY privacy_parameters/ privacy_parameters/
COPY computational_cost/ computational_cost/

# Diretório para o dataset (montado como volume)
RUN mkdir -p dataset

# URL base do Ollama — sobrescrita pelo docker-compose para http://host.docker.internal:11434
ENV OLLAMA_HOST=http://localhost:11434

CMD ["python", "MECAL.py"]
