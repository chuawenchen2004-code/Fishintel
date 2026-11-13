FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Faster install with longer timeout and no cache
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]