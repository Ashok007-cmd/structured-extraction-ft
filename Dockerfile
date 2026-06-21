# Use official PyTorch image with GPU support
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source tree
COPY . .

# Set offline mode environment variables
ENV HF_HUB_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default command: run pytest then start pipeline
CMD ["bash", "-c", "pytest tests/ && python3 scripts/full_pipeline.py"]
