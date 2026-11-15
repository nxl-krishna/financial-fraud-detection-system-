
FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies using a faster package mirror
COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    --default-timeout=120 \
    --retries=5 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# Copy application source code
COPY api/ ./api/
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY reports/ ./reports/

# Create runtime directories
RUN mkdir -p /app/runtime /app/logs

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 8501
