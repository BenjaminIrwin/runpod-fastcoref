# RunPod Serverless Container for FastCoref LingMessCoref
# Based on RunPod's PyTorch base image with CUDA support

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Force eager attention implementation for Longformer compatibility
ENV TRANSFORMERS_ATTENTION_IMPLEMENTATION=eager

# Pre-download the LingMessCoref model during build for fast cold starts
# This bakes the model weights into the container image
# We use a small Python script to properly handle the attention implementation
RUN python -c "
import os
os.environ['TRANSFORMERS_ATTENTION_IMPLEMENTATION'] = 'eager'

# Monkey-patch transformers to use eager attention by default
import transformers.modeling_utils as mu
original_autoset = mu.PreTrainedModel._autoset_attn_implementation
@classmethod
def patched_autoset(cls, config, *args, **kwargs):
    config._attn_implementation = 'eager'
    return config
mu.PreTrainedModel._autoset_attn_implementation = patched_autoset

from fastcoref import LingMessCoref
print('Downloading LingMessCoref model...')
model = LingMessCoref(device='cpu')
print('Model downloaded successfully')
"

# Copy handler
COPY handler.py .

CMD ["python", "-u", "handler.py"]
