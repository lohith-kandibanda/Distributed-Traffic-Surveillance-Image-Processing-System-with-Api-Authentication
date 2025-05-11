FROM traffic-base:latest

WORKDIR /app

COPY helmet_worker/ /app/
COPY models/helmet_detection.pt /app/

CMD ["python", "helmet_worker.py"]
