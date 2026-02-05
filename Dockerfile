FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget unzip libvulkan1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Real-ESRGAN binary
RUN wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip \
    && unzip realesrgan-ncnn-vulkan-*.zip -d /usr/local/bin \
    && chmod +x /usr/local/bin/realesrgan-ncnn-vulkan \
    && rm *.zip

# Download Real-ESRGAN + GFPGAN models with retries
RUN mkdir -p /models /usr/local/bin/models \
    && wget --tries=3 --waitretry=5 -P /usr/local/bin/models \
       https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.bin \
    && wget --tries=3 --waitretry=5 -P /usr/local/bin/models \
       https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.param \
    && wget --tries=3 --waitretry=5 -O /models/GFPGANv1.4.pth \
       https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

RUN pip install --no-cache-dir runpod requests opencv-python-headless numpy gfpgan

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
