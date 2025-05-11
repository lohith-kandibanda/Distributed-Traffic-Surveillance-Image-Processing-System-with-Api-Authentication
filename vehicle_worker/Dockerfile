FROM traffic-base:latest

WORKDIR /app

COPY vehicle_worker/ /app/
COPY models/yolo11m.pt /app/

CMD ["python", "vehicle_worker.py"]
