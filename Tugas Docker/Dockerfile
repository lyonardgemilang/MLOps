# Menggunakan image python yang ringan
FROM python:3.8-slim

# Menginstall make terlebih dahulu
RUN apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/*

# Menyalin seluruh proyek ke direktori /app di dalam container
COPY Makefile_newest /app

# Mengatur direktori kerja ke /app
WORKDIR /app

# Menginstall dependencies dari requirements.txt menggunakan pip dan virtual environment
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Menjalankan perintah make all
CMD ["make", "all"]
