FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Cloud Run injects PORT; default to 8080
ENV PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port $PORT"]
