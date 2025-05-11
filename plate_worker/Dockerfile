FROM traffic-base:latest

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "plate_worker.py"]
