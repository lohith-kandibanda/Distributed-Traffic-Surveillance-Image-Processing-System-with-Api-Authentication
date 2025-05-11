# -----------------------------
# Stage 1: Build and download dependencies
# -----------------------------
FROM python:3.10 AS wheels-builder

WORKDIR /wheels

# Install required system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Download wheels inside container (Linux-compatible)
RUN pip download --no-cache-dir torch torchvision opencv-python easyocr redis pillow aio_pika ultralytics

# -----------------------------
# Stage 2: Final runtime image
# -----------------------------
FROM python:3.10

WORKDIR /app
COPY . /app

# Add OpenCV dependency again in final image
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# ✅ Copy wheels from stage 1
COPY --from=wheels-builder /wheels /wheels

# ✅ Install from wheels
RUN pip install --no-cache-dir /wheels/*

CMD ["python", "worker/worker.py"]

