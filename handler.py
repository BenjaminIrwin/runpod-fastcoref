"""
RunPod Serverless Handler for FastCoref LingMessCoref

Provides coreference resolution using the LingMessCoref model from the fastcoref package.
Returns cluster information with both string representations and character offsets.

API:
    POST https://api.runpod.ai/v2/{endpoint_id}/runsync
    Headers: Authorization: Bearer {api_key}
    Body: {"input": {"texts": ["text1", "text2", ...]}}
    
Response format:
    {
        "results": [
            {
                "clusters": [["We", "our"], ["our coref package", "This package"]],
                "clusters_char_offsets": [[(0, 2), (33, 36)], [(33, 50), (52, 64)]],
                "cluster_logits": {...}  // optional, if return_logits=True
            },
            ...
        ]
    }
"""

import os

# Force eager attention implementation before importing transformers
# This is needed because Longformer doesn't support SDPA
os.environ['TRANSFORMERS_ATTENTION_IMPLEMENTATION'] = 'eager'

# Monkey-patch transformers to use eager attention by default
import transformers.modeling_utils as mu
original_autoset = getattr(mu.PreTrainedModel, '_autoset_attn_implementation', None)
if original_autoset:
    @classmethod
    def patched_autoset(cls, config, *args, **kwargs):
        config._attn_implementation = 'eager'
        return config
    mu.PreTrainedModel._autoset_attn_implementation = patched_autoset

import runpod
import json
from typing import List, Dict, Any, Optional

print("Handler module loading...")

# Global model instance (loaded lazily on first request)
_model = None


def get_model():
    """
    Get or load the LingMessCoref model.
    
    The model is loaded lazily on first request and cached globally.
    Uses CUDA for GPU acceleration.
    """
    global _model
    
    if _model is None:
        print("Loading LingMessCoref model...")
        try:
            from fastcoref import LingMessCoref
            _model = LingMessCoref(device='cuda:0')
            print("LingMessCoref model loaded successfully on CUDA")
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    return _model


def process_prediction(pred, return_logits: bool = False) -> Dict[str, Any]:
    """
    Process a single prediction result into a serializable format.
    
    Args:
        pred: Prediction object from fastcoref
        return_logits: Whether to include logit scores for cluster pairs
        
    Returns:
        Dictionary with clusters and offsets
    """
    result = {
        "clusters": pred.get_clusters(as_strings=True),
        "clusters_char_offsets": pred.get_clusters(as_strings=False),
    }
    
    # Optionally compute logits for each cluster pair
    if return_logits and result["clusters_char_offsets"]:
        cluster_logits = {}
        offsets = result["clusters_char_offsets"]
        
        for cluster_idx, cluster in enumerate(offsets):
            if len(cluster) >= 2:
                # Get logit between first two mentions as representative
                span_i = tuple(cluster[0])
                span_j = tuple(cluster[1])
                try:
                    logit = pred.get_logit(span_i=span_i, span_j=span_j)
                    cluster_logits[f"cluster_{cluster_idx}"] = {
                        "span_i": span_i,
                        "span_j": span_j,
                        "logit": float(logit)
                    }
                except Exception as e:
                    print(f"Warning: Could not get logit for cluster {cluster_idx}: {e}")
        
        result["cluster_logits"] = cluster_logits
    
    return result


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for coreference resolution.
    
    Input parameters:
        texts: List of strings to process (required)
        return_logits: Whether to return logit scores (optional, default: False)
        
    Returns:
        Dictionary with "results" list containing processed predictions
    """
    job_id = job.get('id', 'unknown')
    print(f"Received job: {job_id}")
    
    try:
        job_input = job.get("input", {})
        
        # Get texts - can be a list or a JSON string
        texts = job_input.get("texts", [])
        if isinstance(texts, str):
            try:
                texts = json.loads(texts)
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON in texts field: {str(e)}"}
        
        if not isinstance(texts, list):
            return {"error": "texts must be a list of strings"}
        
        if not texts:
            return {"error": "texts list is empty"}
        
        # Get optional parameters
        return_logits = job_input.get("return_logits", False)
        
        print(f"Processing {len(texts)} text(s)...")
        
        # Load model
        model = get_model()
        
        # Run prediction on all texts
        try:
            preds = model.predict(texts=texts)
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Prediction failed: {str(e)}"}
        
        # Process results
        results = []
        for i, pred in enumerate(preds):
            try:
                result = process_prediction(pred, return_logits=return_logits)
                results.append(result)
            except Exception as e:
                print(f"Error processing prediction {i}: {e}")
                results.append({"error": str(e)})
        
        print(f"Returning {len(results)} result(s)")
        return {"results": results}
        
    except Exception as e:
        print(f"Handler error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# Start the RunPod serverless handler
print("Starting RunPod serverless handler...")
runpod.serverless.start({"handler": handler})
