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

MAX_WAIT_TIME = 60
POLL_INTERVAL = 2
STATIC_DIR = "/app/static/annotated"
STATIC_URL = "http://api_server:8000/static/annotated"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
MASTER_NAME = socket.gethostname()

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def deduplicate_detections(detections, iou_threshold=0.5):
    unique = []
    for det in detections:
        is_duplicate = False
        for u in unique:
            if det['type'] == u['type'] and iou(det['bbox'], u['bbox']) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(det)
    return unique

async def handle_task(message: aio_pika.IncomingMessage):
    async with message.process():
        try:
            task = ast.literal_eval(message.body.decode())
            task_id = task['id']
            logger.info(f"\U0001F3A5 Master received task ID: {task_id}")
            file_data = bytes.fromhex(task['data'])
            task_type = task.get('type', 'video')
            result = await process_input(task_id, file_data, task_type)
            r.set(f"task:{task_id}:result", json.dumps(result))
            r.set(f"task:{task_id}:status", "done")
            logger.info(f"\u2705 Master saved result for Task ID: {task_id}")
        except Exception as e:
            logger.error(f"\u274C Master failed to process task: {e}")

async def process_input(task_id, file_bytes, task_type):
    temp_suffix = ".jpg" if task_type == "image" else ".mp4"
    sent_frames = []
    frame_detections = {}
    full_frames = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix) as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    channel = await connection.channel()
    await channel.declare_queue(VEHICLE_QUEUE, durable=True)
    await channel.declare_queue(PLATE_QUEUE, durable=True)
    await channel.declare_queue(HELMET_QUEUE, durable=True)

    if task_type == "image":
        frame = cv2.imread(temp_path)
        frame_no = 1
        full_frames.append((frame_no, frame.copy()))
        await send_frame_to_workers(task_id, frame_no, frame, channel)
        sent_frames.append(frame_no)
    else:
        cap = cv2.VideoCapture(temp_path)
        frame_no = 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_no in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"⚠️ Failed to read frame {frame_no}")
                continue

            full_frames.append((frame_no, frame.copy()))
            await send_frame_to_workers(task_id, frame_no, frame, channel)
            sent_frames.append(frame_no)
            frame_no += 1
        cap.release()

    await connection.close()
    logger.info(f"\U0001F4E4 Master sent {len(sent_frames)} frames to workers.")

    vehicle_results = await collect_results(task_id, "vehicle", sent_frames)
    plate_results = await collect_results(task_id, "plate", sent_frames)
    helmet_results = await collect_results(task_id, "helmet", sent_frames)

    for res in vehicle_results:
        frame_detections.setdefault(res['frame_no'], {})['vehicles'] = res.get('vehicles', [])
    for res in plate_results:
        frame_detections.setdefault(res['frame_no'], {})['plates'] = res.get('plates', [])
    for res in helmet_results:
        frame_detections.setdefault(res['frame_no'], {})['helmet_violations'] = res.get('helmet_violations', [])

    os.makedirs(STATIC_DIR, exist_ok=True)
    timestamp = int(datetime.utcnow().timestamp())
    annotated_filename = f"{task_id}_{timestamp}.{'jpg' if task_type == 'image' else 'mp4'}"
    annotated_path = os.path.join(STATIC_DIR, annotated_filename)

    if task_type == "image":
        frame_no, frame = full_frames[0]
        detections = frame_detections.get(frame_no, {})
        annotated = draw_annotations(
            frame,
            detections.get("vehicles", []),
            detections.get("plates", []),
            detections.get("helmet_violations", []),
            frame_no
        )
        cv2.imwrite(annotated_path, annotated)
        logger.info(f"🖼 Annotated image saved to: {annotated_path}")
    else:
        height, width = full_frames[0][1].shape[:2]
        fps = cv2.VideoCapture(temp_path).get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(
            annotated_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width + 100, height + 100)
        )
        for frame_no, frame in full_frames:
            detections = frame_detections.get(frame_no, {})
            annotated = draw_annotations(
                frame,
                detections.get("vehicles", []),
                detections.get("plates", []),
                detections.get("helmet_violations", []),
                frame_no
            )
            out.write(annotated)
        out.release()
        logger.info(f"🎥 Annotated video saved to: {annotated_path}")

    os.remove(temp_path)

    all_vehicles = [v for detections in frame_detections.values() for v in detections.get("vehicles", [])]
    unique_vehicles = deduplicate_detections(all_vehicles)

    vehicle_types = {}
    for v in unique_vehicles:
        vtype = v["type"]
        vehicle_types[vtype] = vehicle_types.get(vtype, 0) + 1

    all_plates = [p for detections in frame_detections.values() for p in detections.get("plates", [])]
    all_violations = [h for detections in frame_detections.values() for h in detections.get("helmet_violations", [])]

    unique_plates = list({p["plate_text"]: p for p in all_plates}.values())
    helmet_violations_with_plates = []
    for hv in all_violations:
        best_match = None
        for plate in all_plates:
            if iou(hv["bbox"], plate["bbox"]) > 0.5:
                best_match = plate["plate_text"]
                break
        helmet_violations_with_plates.append({
            "plate": best_match or "Unknown",
            "bbox": hv["bbox"]
        })

    return {
        "total_frames_processed": len(sent_frames),
        "vehicle_count": len(unique_vehicles),
        "vehicle_types": vehicle_types,
        "license_plates": [p["plate_text"] for p in unique_plates],
        "helmet_violations": helmet_violations_with_plates,
        "type": task_type,
        "annotated_url": f"{STATIC_URL}/{annotated_filename}"
    }


import cv2

def draw_label_with_bg(image, text, x, y, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, text_color=(0, 0, 0), bg_color=(0, 255, 0), thickness=1, padding=4):
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    cv2.rectangle(image, (x, y - text_h - padding), (x + text_w + padding*2, y), bg_color, -1)
    cv2.putText(image, text, (x + padding, y - padding), font, font_scale, text_color, thickness, cv2.LINE_AA)

def draw_annotations(frame, vehicles, plates, helmet_violations, frame_no):
    top_pad = 50
    bottom_pad = 50
    left_pad = 50   # add left padding
    right_pad = 50  # add right padding

    padded_frame = cv2.copyMakeBorder(
        frame, top_pad, bottom_pad, left_pad, right_pad,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # Frame number label
    draw_label_with_bg(padded_frame, f"Frame {frame_no}", x=10 + left_pad, y=top_pad - 10, bg_color=(255, 255, 0))

    for v in vehicles:
        x1, y1, x2, y2 = [coord + left_pad if i % 2 == 0 else coord + top_pad for i, coord in enumerate(v["bbox"])]
        cv2.rectangle(padded_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        draw_label_with_bg(padded_frame, f"Vehicle: {v['type']}", x1, max(y1 - 5, top_pad))

    for p in plates:
        x1, y1, x2, y2 = [coord + left_pad if i % 2 == 0 else coord + top_pad for i, coord in enumerate(p["bbox"])]
        cv2.rectangle(padded_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        draw_label_with_bg(padded_frame, f"Plate: {p['plate_text']}", x1, y2 + 20, bg_color=(0, 255, 255))

    for hv in helmet_violations:
        x1, y1, x2, y2 = [coord + left_pad if i % 2 == 0 else coord + top_pad for i, coord in enumerate(hv["bbox"])]
        cv2.rectangle(padded_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label_y = y1 - 10 if y1 - 10 > top_pad else y2 + 20
        draw_label_with_bg(padded_frame, "No Helmet", x1, label_y, bg_color=(0, 0, 255))

    return padded_frame


async def send_frame_to_workers(task_id, frame_no, frame, channel):
    _, buffer = cv2.imencode('.jpg', frame)
    frame_hex = buffer.tobytes().hex()
    frame_task = {
        "parent_id": task_id,
        "frame_no": frame_no,
        "timestamp": datetime.utcnow().isoformat(),
        "frame_data": frame_hex
    }
    for queue, task_type in [(VEHICLE_QUEUE, "vehicle"), (PLATE_QUEUE, "plate"), (HELMET_QUEUE, "helmet")]:
        await channel.default_exchange.publish(
            aio_pika.Message(body=str({**frame_task, "type": task_type}).encode()),
            routing_key=queue
        )

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
            logger.warning(f"\u26A0\ufe0f Timeout for {task_type} result Frame {frame_no}")
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
    logger.info(f"\U0001F680 Master [{MASTER_NAME}] ready and listening for tasks...")
    return connection

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    conn = loop.run_until_complete(main())
    loop.run_forever()
