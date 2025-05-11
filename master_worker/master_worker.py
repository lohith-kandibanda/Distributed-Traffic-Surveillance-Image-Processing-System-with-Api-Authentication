import aio_pika
import redis
import cv2
import tempfile
import os
import asyncio
import ast
import socket
import json
import logging
from datetime import datetime

RABBITMQ_URL = "amqp://guest:guest@rabbitmq/"
MAIN_QUEUE = "task_queue"
VEHICLE_QUEUE = "vehicle_queue"
PLATE_QUEUE = "plate_queue"
HELMET_QUEUE = "helmet_task_queue"

REDIS_HOST = "redis"
REDIS_PORT = 6379

FRAME_SKIP = 5
MAX_WAIT_TIME = 30
POLL_INTERVAL = 2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

MASTER_NAME = socket.gethostname()

async def handle_task(message: aio_pika.IncomingMessage):
    async with message.process():
        task = None
        try:
            task = ast.literal_eval(message.body.decode())
            task_id = task['id']
            logger.info(f"🎥 Master received video Task ID: {task_id}")

            video_data = bytes.fromhex(task['data'])
            final_result = await process_video(task_id, video_data)

            r.set(f"task:{task_id}:result", json.dumps(final_result))
            r.set(f"task:{task_id}:status", "done")

            logger.info(f"✅ Master saved final result for Task ID: {task_id}")

        except Exception as e:
            logger.error(f"❌ Master failed to process task: {e}")

async def process_video(task_id, video_bytes):
    sent_frames = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_vid:
        temp_vid.write(video_bytes)
        temp_path = temp_vid.name

    cap = cv2.VideoCapture(temp_path)
    frame_no = 0

    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    channel = await connection.channel()

    await channel.declare_queue(VEHICLE_QUEUE, durable=True)
    await channel.declare_queue(PLATE_QUEUE, durable=True)
    await channel.declare_queue(HELMET_QUEUE, durable=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % FRAME_SKIP != 0:
            continue

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        frame_hex = frame_bytes.hex()

        frame_task = {
            "parent_id": task_id,
            "frame_no": frame_no,
            "timestamp": datetime.utcnow().isoformat(),
            "frame_data": frame_hex
        }

        await channel.default_exchange.publish(
            aio_pika.Message(body=str({**frame_task, "type": "vehicle"}).encode()),
            routing_key=VEHICLE_QUEUE
        )
        await channel.default_exchange.publish(
            aio_pika.Message(body=str({**frame_task, "type": "plate"}).encode()),
            routing_key=PLATE_QUEUE
        )
        await channel.default_exchange.publish(
            aio_pika.Message(body=str({**frame_task, "type": "helmet"}).encode()),
            routing_key=HELMET_QUEUE
        )

        sent_frames.append(frame_no)

    cap.release()
    os.remove(temp_path)
    await connection.close()

    logger.info(f"📤 Master sent {len(sent_frames)} frames to workers.")

    vehicle_results = await collect_results(task_id, "vehicle", sent_frames)
    plate_results = await collect_results(task_id, "plate", sent_frames)
    helmet_results = await collect_results(task_id, "helmet", sent_frames)

    vehicle_types = {}
    for vlist in vehicle_results:
        for v in vlist.get("vehicles", []):
            vtype = v["type"]
            vehicle_types[vtype] = vehicle_types.get(vtype, 0) + 1

    unique_plates = {}
    for plist in plate_results:
        for p in plist.get("plates", []):
            plate_text = p["plate_text"]
            unique_plates[plate_text] = p

    helmet_violations_with_plates = []
    for hlist in helmet_results:
        for hv in hlist.get("helmet_violations", []):
            for p in plate_results:
                for plate in p.get("plates", []):
                    helmet_violations_with_plates.append({
                        "plate": plate.get("plate_text"),
                        "violation": hv.get("violation"),
                        "bbox": hv.get("bbox")
                    })

    final_result = {
        "total_frames_processed": len(sent_frames),
        "vehicle_count": sum(vehicle_types.values()),
        "vehicle_types": vehicle_types,
        "license_plates": list(unique_plates.keys()),
        "helmet_plate_violations": helmet_violations_with_plates
    }

    return final_result

async def collect_results(parent_id, task_type, frame_list):
    results = []
    for frame_no in frame_list:
        key = f"{parent_id}:{task_type}:{frame_no}"
        waited = 0

        while waited < MAX_WAIT_TIME:
            result_json = r.get(key)
            if result_json:
                result = json.loads(result_json)
                results.append(result)
                r.delete(key)
                break
            await asyncio.sleep(POLL_INTERVAL)
            waited += POLL_INTERVAL

        if waited >= MAX_WAIT_TIME:
            logger.warning(f"⚠️ Timeout for {task_type} result Frame {frame_no}")

    return results

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
    queue = await channel.declare_queue(MAIN_QUEUE, durable=True)
    await queue.consume(handle_task)
    logger.info(f"🚀 Master [{MASTER_NAME}] ready and listening for tasks...")
    return connection

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    conn = loop.run_until_complete(main())
    loop.run_forever()
