# Usa Python 3.11 slim para menor tamaño
FROM python:3.11-slim

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements primero para aprovechar cache de Docker
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Crear directorios necesarios
RUN mkdir -p data/tmp data/vector_stores

# Copiar código de la aplicación
COPY . .

# Crear usuario no-root por seguridad
RUN useradd -m -u 1000 streamlituser && \
    chown -R streamlituser:streamlituser /app
USER streamlituser

# Exponer puerto
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Variables de entorno
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Comando principal - cambiar a main.py
CMD ["streamlit", "run", "main.py", "--server.address", "0.0.0.0", "--server.port", "8501"]