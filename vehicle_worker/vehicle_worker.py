# vehicle_worker/vehicle_worker.py

import aio_pika
import redis
import ast
import torch
import cv2
import numpy as np
import socket
import logging
import json
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import asyncio

# ---------------------------
# Configuration
# ---------------------------

RABBITMQ_URL = "amqp://guest:guest@rabbitmq/"
VEHICLE_QUEUE = "vehicle_queue"

REDIS_HOST = "redis"
REDIS_PORT = 6379

CONFIDENCE_THRESHOLD = 0.25

# ---------------------------
# Initialization
# ---------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
WORKER_NAME = socket.gethostname()

device = "cuda" if torch.cuda.is_available() else "cpu"
vehicle_model = YOLO("/models/yolo11m.pt").to(device)

# ---------------------------
# Task Handler
# ---------------------------

async def handle_task(message: aio_pika.IncomingMessage):
    async with message.process(requeue=True):
        task = None
        try:
            task = ast.literal_eval(message.body.decode())
            logger.info(f"🚗 Vehicle Worker [{WORKER_NAME}] processing Frame {task['frame_no']} of Task {task['parent_id']}")

            frame_data = bytes.fromhex(task['frame_data'])
            image = np.array(Image.open(BytesIO(frame_data)).convert("RGB"))

            result = process_frame(task['parent_id'], task['frame_no'], image)

            frame_key = f"{task['parent_id']}:vehicle:{task['frame_no']}"
            r.set(frame_key, json.dumps(result))

            logger.info(f"✅ Vehicle Worker [{WORKER_NAME}] finished Frame {task['frame_no']}")

        except Exception as e:
            logger.error(f"❌ Vehicle Worker [{WORKER_NAME}] failed Frame {task['frame_no'] if task else 'UNKNOWN'}: {e}")
            raise e

# ---------------------------
# Frame Processor
# ---------------------------

def process_frame(parent_task_id, frame_no, image):
    vehicle_preds = vehicle_model.predict(source=image, conf=CONFIDENCE_THRESHOLD)[0]
    vehicle_boxes = []

    for det in vehicle_preds.boxes:
        cls = int(det.cls.item())
        name = vehicle_model.model.names[cls]
        if name.lower() in ["car", "motorbike", "truck", "bus", "bicycle"]:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            vehicle_boxes.append({
                "type": name,
                "bbox": [x1, y1, x2, y2]
            })

    return {
        "frame_no": frame_no,
        "vehicle_count": len(vehicle_boxes),
        "vehicles": vehicle_boxes
    }

# ---------------------------
# Worker Entry with Retry
# ---------------------------

async def wait_for_rabbitmq():
    while True:
        try:
            connection = await aio_pika.connect_robust(RABBITMQ_URL)
            return connection
        except Exception:
            logger.warning("Waiting for RabbitMQ to be ready...")
            await asyncio.sleep(5)

async def main():
    connection = await wait_for_rabbitmq()
    channel = await connection.channel()
    await channel.declare_queue(VEHICLE_QUEUE, durable=True)
    queue = await channel.declare_queue(VEHICLE_QUEUE, durable=True)
    await queue.consume(handle_task)
    logger.info(f"🚀 Vehicle Worker [{WORKER_NAME}] is running and waiting for frames...")
    return connection

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    conn = loop.run_until_complete(main())
    loop.run_forever()
