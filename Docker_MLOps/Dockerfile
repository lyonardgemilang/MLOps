FROM python:3.8-slim
RUN apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/*
COPY . /app
WORKDIR /app
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt
CMD ["make", "all"]