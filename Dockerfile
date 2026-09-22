FROM ghcr.io/leejet/stable-diffusion.cpp:master-cuda

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ca-certificates \
        aria2 && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODEL_REPO=KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF \
    DIFFUSION_FILE=qwen-image-2.1-Q4_K_M.gguf \
    LLM_REPO=Qwen/Qwen3-VL-8B-Instruct-GGUF \
    LLM_FILE=Qwen3VL-8B-Instruct-Q4_K_M.gguf \
    VAE_FILE=vae/qwen_image_2.1_vae_bf16.safetensors \
    SD_SERVER_BIN=/sd-server \
    SD_PORT=1234 \
    SD_STEPS=25 \
    SD_CFG=6 \
    SD_WIDTH=1024 \
    SD_HEIGHT=1024 \
    SD_OFFLOAD=0

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

WORKDIR /app
COPY handler.py /app/handler.py
COPY test_image.b64 /app/test_image.b64

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD ["python", "-u", "/app/handler.py"]
