# RunPod Serverless Container for FastCoref LingMessCoref
# Based on RunPod's PyTorch base image with CUDA support

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Force eager attention implementation for Longformer compatibility
ENV TRANSFORMERS_ATTENTION_IMPLEMENTATION=eager

# Copy files
COPY download_model.py handler.py ./

# Pre-download the LingMessCoref model during build for fast cold starts
# This bakes the model weights into the container image
RUN python download_model.py

CMD ["python", "-u", "handler.py"]
