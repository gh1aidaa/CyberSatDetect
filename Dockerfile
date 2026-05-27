FROM python:3.11-slim

WORKDIR /app

# Keep Python output unbuffered for logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY . .

# Render / most platforms inject PORT
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.app.api:app --host 0.0.0.0 --port ${PORT}"]

