# RunPod Serverless Container for FastCoref LingMessCoref
# Using official PyTorch runtime image (3.2GB vs 7GB+ for RunPod images)

FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

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
