import aio_pika
import redis
import ast
import socket
import logging
import asyncio
import json
import cv2
import numpy as np
from io import BytesIO
from ultralytics import YOLO
from PIL import Image
import os

# ---------------------------
# Configuration
# ---------------------------
RABBITMQ_URL = "amqp://guest:guest@rabbitmq/"
HELMET_QUEUE = "helmet_task_queue"
REDIS_HOST = "redis"
REDIS_PORT = 6379
CONFIDENCE_THRESHOLD = 0.3
ANNOTATION_DIR = "static/annotated"

# ---------------------------
# Initialization
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
WORKER_NAME = socket.gethostname()
helmet_model = YOLO("helmet_detection.pt")

# Ensure annotation directory exists
os.makedirs(ANNOTATION_DIR, exist_ok=True)

# ---------------------------
# Task Handler
# ---------------------------
async def handle_task(message: aio_pika.IncomingMessage):
    async with message.process(requeue=True):
        task = None
        try:
            task = ast.literal_eval(message.body.decode())
            parent_id = task['parent_id']
            frame_no = task['frame_no']

            logger.info(f"🛡️ Helmet Worker [{WORKER_NAME}] processing Frame {frame_no} for Task {parent_id}")

            frame_data = bytes.fromhex(task['frame_data'])
            frame = np.array(Image.open(BytesIO(frame_data)).convert('RGB'))

            result = process_helmet_frame(frame, parent_id, frame_no)

            key = f"{parent_id}:helmet:{frame_no}"
            r.set(key, json.dumps(result))

            logger.info(f"✅ Helmet Worker [{WORKER_NAME}] saved result for Frame {frame_no}")

        except Exception as e:
            logger.error(f"❌ Helmet Worker [{WORKER_NAME}] failed processing Frame {task['frame_no'] if task else 'UNKNOWN'}: {e}")
            raise e

# ---------------------------
# Helmet Detection Logic
# ---------------------------
def process_helmet_frame(frame, task_id, frame_no):
    helmet_violations = []
    annotated_frame = frame.copy()
    annotated_path = None

    try:
        detections = helmet_model.predict(source=frame, conf=CONFIDENCE_THRESHOLD)[0]

        for det in detections.boxes:
            cls = int(det.cls.item())
            conf = round(det.conf.item(), 3)
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())

            if cls == 2:  # WithoutHelmet
                helmet_violations.append({
                    "violation": "No Helmet",
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf
                })
                color = (0, 0, 255)
                label = f"No Helmet ({conf})"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, max(10, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            elif cls == 1:  # WithHelmet (optional)
                color = (0, 255, 0)
                label = f"With Helmet ({conf})"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(annotated_frame, label, (x1, max(10, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Ignore class 0 (Plate)

        if helmet_violations:
            annotated_path = os.path.join(ANNOTATION_DIR, f"{task_id}_{frame_no}_helmet.jpg")
            cv2.imwrite(annotated_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

    except Exception as e:
        logger.error(f"❌ Helmet detection failed: {e}")

    return {
        "helmet_violations": helmet_violations,
        "violation_count": len(helmet_violations),
        "frame_no": frame_no,
        "annotated_image": annotated_path
    }


# ---------------------------
# Worker Startup
# ---------------------------
async def wait_for_rabbitmq():
    while True:
        try:
            connection = await aio_pika.connect_robust(RABBITMQ_URL)
            return connection
        except aio_pika.exceptions.AMQPConnectionError:
            logger.warning("Waiting for RabbitMQ to be ready...")
            await asyncio.sleep(5)

async def main():
    connection = await wait_for_rabbitmq()
    channel = await connection.channel()
    queue = await channel.declare_queue(HELMET_QUEUE, durable=True)
    await queue.consume(handle_task)
    logger.info(f"🚀 Helmet Worker [{WORKER_NAME}] is waiting for helmet tasks...")
    return connection

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    conn = loop.run_until_complete(main())
    loop.run_forever()
