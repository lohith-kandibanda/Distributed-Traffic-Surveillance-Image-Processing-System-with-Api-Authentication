# 🚦 Distributed Traffic Surveillance System with API Authentication

An end-to-end, fault-tolerant distributed system for real-time **traffic surveillance**, utilizing **YOLOv11** for object detection, **PaddleOCR** for license plate recognition, and an **asynchronous master-worker architecture** built using **FastAPI**, **RabbitMQ**, and **Redis**. The project provides a full-stack experience with a **Streamlit-based UI**, robust backend API, Docker-based deployment, and fault tolerance mechanisms.

---

## 📁 Project Structure

```
.
├── api_server/           # FastAPI backend handling uploads & authentication
├── master_worker/        # Orchestrates frame splitting and task delegation
├── vehicle_worker/       # Detects vehicles using YOLOv11
├── plate_worker/         # Extracts license plates using PaddleOCR
├── helmet_worker/        # Detects helmet violations using YOLOv9n
├── frontend/             # Streamlit-based UI to upload/view results
├── static/               # Shared folder for annotated results
├── models/               # Stores YOLO weights
├── docker-compose.yml    # Multi-container orchestration
└── README.md             # Documentation (this file)
```

---

## 🚀 Features

* 🎥 Frame-by-frame video/image analysis
* 🧠 YOLO-based vehicle & helmet detection
* 🔍 OCR-based number plate extraction
* 🐇 Asynchronous RabbitMQ task queues
* 🧾 Redis for task/result storage & rate limiting
* 💽 Streamlit-based frontend UI
* 🐳 Docker Compose for seamless deployment
* ♻️ Fault tolerance & graceful recovery

---

## ⚙️ How to Run

### 1️⃣ Prerequisites

* [Docker](https://www.docker.com/products/docker-desktop)
* [Docker Compose](https://docs.docker.com/compose/)
* 8GB+ RAM recommended for parallel processing

### 2️⃣ Clone the Repository

```bash
git clone https://github.com/lohith-kandibanda/Distributed-Traffic-Surveillance-Image-Processing-System-with-Api-Authentication.git
cd Distributed-Traffic-Surveillance-Image-Processing-System-with-Api-Authentication
```

### 3️⃣ Build and Start the System

```bash
docker-compose up --build
```

### 4️⃣ Access the Interfaces

* **Streamlit UI**: [http://localhost:8501](http://localhost:8501)
* **FastAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **RabbitMQ Dashboard**: [http://localhost:15672](http://localhost:15672) (guest/guest)
* **Redis (CLI)**: `docker exec -it <redis_container> redis-cli`

---

## 🛡️ API Security

All API requests must include:

```http
X-API-Key: traffic123
```

Rate limit: **10 requests/minute per key** (via Redis)

---

## 🧪 Demo Flow

1. Upload an image/video via UI or API
2. Master receives and splits task into frames
3. Each frame is sent to:

   * `vehicle_worker`
   * `plate_worker`
   * `helmet_worker`
4. Workers store results in Redis
5. Master fetches, annotates, and saves final output
6. UI/API fetches results via `/result/{task_id}`

---

## 📦 Fault Tolerance

* Master retries Redis & RabbitMQ connection until ready
* Workers write to Redis; if they crash, tasks remain in RabbitMQ
* Annotated outputs generated even if partial results available

To demonstrate:

```bash
docker-compose stop helmet_worker
# Upload a task and watch the rest of the system work
docker-compose start helmet_worker
```

---

## 📊 Tech Stack

* **YOLOv11m / YOLOv9n** – Object Detection
* **PaddleOCR** – Number Plate Recognition
* **FastAPI** – REST API
* **Redis** – Storage, Status, Rate Limiting
* **RabbitMQ** – Asynchronous Messaging
* **OpenCV** – Frame manipulation and video handling
* **aio-pika** – Async messaging
* **Docker Compose** – Multi-container deployment
* **Streamlit** – Web UI

---

## 🤝 Author

Built by **Lohith Kandibanda** – for scalable, secure, and intelligent traffic monitoring 🚗🟍️🚵️

Feel free to ⭐ the repo and contribute!

---

## 🦾 License

MIT License
