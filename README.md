# FastCoref LingMessCoref on RunPod Serverless

Deploy [fastcoref](https://github.com/shon-otmazgin/fastcoref)'s LingMessCoref model on RunPod Serverless for GPU-accelerated coreference resolution.

## Features

- **Serverless GPU**: Pay only when processing, scales to zero
- **Fast cold starts**: Model baked into container (~10-20s cold start)
- **Batch processing**: Process multiple texts in a single request
- **Character offsets**: Returns both text and character positions for clusters

## Quick Start

### 1. Deploy to RunPod

#### Option A: Use GitHub Actions (Recommended)

1. Fork/clone this repo to your GitHub account
2. Push to trigger the build workflow
3. Image will be pushed to `ghcr.io/<your-username>/fastcoref:latest`

#### Option B: Build Locally

```bash
docker build -t fastcoref .
docker tag fastcoref ghcr.io/<your-username>/fastcoref:latest
docker push ghcr.io/<your-username>/fastcoref:latest
```

### 2. Create RunPod Serverless Endpoint

1. Go to [RunPod Console](https://console.runpod.io/serverless)
2. Click "New Endpoint"
3. Configure:
   - **Container Image**: `ghcr.io/<your-username>/fastcoref:latest`
   - **GPU**: 16GB L4 or RTX 4000 Ada (cheapest options that fit the model)
   - **Max Workers**: 1-3 depending on expected load
   - **Idle Timeout**: 5-10 seconds (for cost optimization)
4. Save and note your Endpoint ID

### 3. Use the Client

```bash
pip install runpod
```

```python
from client import FastCorefClient

# Set your API key (or use RUNPOD_API_KEY env var)
client = FastCorefClient(
    endpoint_id="your_endpoint_id",
    api_key="your_runpod_api_key"
)

# Single text
result = client.predict("We are happy to see you using our package. This package is fast!")
print(result.clusters)
# [['We', 'our'], ['our package', 'This package']]

print(result.clusters_char_offsets)
# [[(0, 2), (33, 36)], [(33, 50), (52, 64)]]

# Batch processing
results = client.predict_batch([
    "Barack Obama was president. He served two terms.",
    "The company announced profits. It exceeded expectations."
])
```

## API Reference

### Request

```json
POST https://api.runpod.ai/v2/{endpoint_id}/runsync
Headers: Authorization: Bearer {api_key}

{
    "input": {
        "texts": ["text1", "text2", ...],
        "return_logits": false
    }
}
```

### Response

```json
{
    "results": [
        {
            "clusters": [["We", "our"], ["our coref package", "This package"]],
            "clusters_char_offsets": [[[0, 2], [33, 36]], [[33, 50], [52, 64]]],
            "cluster_logits": {}
        }
    ]
}
```

## Cost Estimate

| Usage | Cost |
|-------|------|
| GPU (16GB L4) | ~$0.24-0.34/hr |
| 100 requests/day, ~200ms each | ~$0.50-1.00/month |
| 1000 requests/day | ~$5-10/month |

Compared to SageMaker real-time GPU endpoint: ~$380/month (always running)

## Cold Start Optimization

The Dockerfile pre-downloads the model during build to minimize cold starts:

```dockerfile
RUN python -c "from fastcoref import LingMessCoref; LingMessCoref(device='cpu')"
```

Expected cold start: **10-20 seconds** with RunPod FlashBoot.

To eliminate cold starts entirely, configure at least 1 "Active Worker" in RunPod settings.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RUNPOD_API_KEY` | Your RunPod API key |
| `RUNPOD_FASTCOREF_ENDPOINT_ID` | Endpoint ID (default: `vd145wze17hpc0`) |

## Development

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Test handler locally (requires GPU)
python handler.py --rp_serve_api
```

### Docker Build

```bash
docker build -t fastcoref .
docker run --gpus all -p 8000:8000 fastcoref
```

## License

MIT License

## References

- [fastcoref](https://github.com/shon-otmazgin/fastcoref) - The underlying coreference model
- [RunPod Python SDK](https://github.com/runpod/runpod-python) - Official RunPod library
- [RunPod Serverless Docs](https://docs.runpod.io/serverless)
