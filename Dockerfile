FROM python:3.12-slim

# Optional OCR support: uncomment to handle scanned documents
# RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY demo ./demo

# Local-first by default: no document leaves the container
ENV ETD_PROVIDER=none
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
