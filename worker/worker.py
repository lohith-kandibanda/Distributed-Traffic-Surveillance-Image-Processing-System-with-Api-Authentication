import aio_pika
import redis
import ast
import torch
import easyocr
import cv2
import numpy as np
import logging
import socket
import json
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import tempfile
import os

# ---------------------------
# Configuration
# ---------------------------

RABBITMQ_URL = "amqp://guest:guest@rabbitmq/"
QUEUE_NAME = "task_queue"
REDIS_HOST = "redis"
REDIS_PORT = 6379

# ---------------------------
# Initialize
# ---------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
WORKER_NAME = socket.gethostname()

# Load Models
vehicle_model = YOLO("../models/yolo11m.pt")
helmet_model = YOLO("../models/helmet_detection.pt")
plate_reader = easyocr.Reader(['en'], gpu=False)

# ---------------------------
# Task Handler
# ---------------------------

async def handle_task(message: aio_pika.IncomingMessage):
    async with message.process(requeue=True):
        task = None
        try:
            task = ast.literal_eval(message.body.decode())
            logger.info(f"🚀 Worker [{WORKER_NAME}] started processing Task ID: {task['id']}")

            if task.get('type') == 'image':
                image_data = bytes.fromhex(task['data'])
                image = np.array(Image.open(BytesIO(image_data)).convert("RGB"))
                result = process_single_image(image)

            elif task.get('type') == 'video':
                video_data = bytes.fromhex(task['data'])
                result = process_video_stream(video_data)

            else:
                raise Exception("Unknown task type")

            # Attach worker info
            result["worker"] = WORKER_NAME

            # Store result
            r.set(f"task:{task['id']}:result", json.dumps(result))
            r.set(f"task:{task['id']}:status", "done")

            logger.info(f"✅ Worker [{WORKER_NAME}] successfully completed Task ID: {task['id']}")

        except Exception as e:
            logger.error(f"❌ Worker [{WORKER_NAME}] failed on Task {task['id'] if task else 'UNKNOWN'}: {e}")
            raise e

# ---------------------------
# Image Processor
# ---------------------------

def process_single_image(image):
    vehicles, plates, helmets = run_detections(image)
    return {
        "vehicle_count": len(vehicles),
        "vehicles": vehicles,
        "license_plates": plates,
        "helmet_violations": helmets
    }

# ---------------------------
# Video Processor
# ---------------------------

def process_video_stream(video_bytes):
    vehicle_results = []
    plate_results = []
    helmet_violations = []

    # Write video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_bytes)
        temp_filename = temp_video.name

    cap = cv2.VideoCapture(temp_filename)
    frame_no = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1

        # Process every 5th frame
        if frame_no % 5 != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vehicles, plates, helmets = run_detections(frame_rgb)

        vehicle_results.extend(vehicles)
        plate_results.extend(plates)
        helmet_violations.extend(helmets)

    cap.release()
    os.remove(temp_filename)

    return {
        "total_frames_processed": frame_no,
        "vehicle_count": len(vehicle_results),
        "vehicles": vehicle_results,
        "license_plates": plate_results,
        "helmet_violations": helmet_violations
    }

# ---------------------------
# Detection Runner
# ---------------------------

def run_detections(image):
    # Vehicle detection
    vehicle_preds = vehicle_model.predict(source=image, conf=0.25)[0]
    vehicle_boxes = []
    for det in vehicle_preds.boxes:
        cls = int(det.cls.item())
        name = vehicle_model.model.names[cls]
        if name in ["car", "motorbike", "truck", "bus", "bicycle"]:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            vehicle_boxes.append({"type": name, "bbox": [x1, y1, x2, y2]})

    # License Plate detection
    plates = plate_reader.readtext(image)
    plate_texts = [{"plate_text": text[1]} for text in plates]

    # Helmet violation detection
    helmet_preds = helmet_model.predict(source=image, conf=0.25)[0]
    helmet_violations = []
    for det in helmet_preds.boxes:
        cls = int(det.cls.item())
        name = helmet_model.model.names[cls]
        if name.lower() in ["no_helmet", "without_helmet"]:  # adjust based on your class name
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            helmet_violations.append({"bbox": [x1, y1, x2, y2]})

    return vehicle_boxes, plate_texts, helmet_violations

# ---------------------------
# Worker Entry
# ---------------------------

async def main():
    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    channel = await connection.channel()
    queue = await channel.declare_queue(QUEUE_NAME, durable=True)
    await queue.consume(handle_task)
    logger.info(f"🚀 Worker [{WORKER_NAME}] is running and waiting for tasks...")
    return connection

if __name__ == "__main__":
    import asyncio
    loop = asyncio.get_event_loop()
    conn = loop.run_until_complete(main())
    loop.run_forever()
