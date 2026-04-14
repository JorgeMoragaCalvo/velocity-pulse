FROM python:3.9-slim-bullseye

# System dependencies for OpenCV + video
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model weights at build time so container works offline
RUN python -c "\
import torch; \
torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', pretrained=True, trust_repo=True); \
print('MiDaS downloaded')"

RUN python -c "\
from ultralytics import YOLO; \
YOLO('yolov8n-seg.pt'); \
print('YOLOv8n-seg downloaded')"

# Copy source
COPY . .

# Mount points for external data and outputs
RUN mkdir -p /app/TEST_VEL /app/outputs /app/config

VOLUME ["/app/TEST_VEL", "/app/outputs", "/app/config"]

ENTRYPOINT ["python", "main.py"]
CMD ["--method", "fusion", "--input", "/app/TEST_VEL"]
