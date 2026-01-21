"""
FastCoref LingMessCoref RunPod Client

Client for calling the FastCoref LingMessCoref service on RunPod Serverless.
Uses the official runpod Python library (https://github.com/runpod/runpod-python).

Usage:
    from client import FastCorefClient
    
    client = FastCorefClient(endpoint_id="your_endpoint_id")
    
    # Single text (synchronous)
    result = client.predict("We are happy to see you using our package. This package is fast!")
    print(result.clusters)  # [['We', 'our'], ['our package', 'This package']]
    
    # Batch processing
    results = client.predict_batch(["text1", "text2", "text3"])
    
    # Async (returns job object for polling)
    job = client.predict_async("Some text here")
    print(job.status())
    print(job.output())
"""

import os
from typing import List, Optional, Tuple
from dataclasses import dataclass

import runpod


@dataclass
class CorefMention:
    """A mention detected by the coreference model."""
    text: str
    start: int  # Character offset
    end: int    # Character offset


@dataclass
class CorefResult:
    """Result from FastCoref coreference resolution."""
    clusters: List[List[str]]  # Cluster strings
    clusters_char_offsets: List[List[Tuple[int, int]]]  # Character offsets
    cluster_logits: Optional[dict] = None  # Optional logit scores
    
    @property
    def mention_count(self) -> int:
        """Total number of mentions across all clusters."""
        return sum(len(c) for c in self.clusters)
    
    @property
    def cluster_count(self) -> int:
        """Number of coreference clusters."""
        return len(self.clusters)
    
    def get_mentions(self, original_text: str) -> List[List[CorefMention]]:
        """
        Convert clusters to CorefMention objects with text extracted from original.
        
        Args:
            original_text: The original text that was analyzed
            
        Returns:
            List of clusters, each containing CorefMention objects
        """
        result = []
        for cluster_offsets in self.clusters_char_offsets:
            cluster_mentions = []
            for start, end in cluster_offsets:
                text = original_text[start:end] if 0 <= start < end <= len(original_text) else ""
                cluster_mentions.append(CorefMention(text=text, start=start, end=end))
            result.append(cluster_mentions)
        return result


class FastCorefClient:
    """
    Client for RunPod serverless FastCoref endpoint.
    
    Uses the official runpod Python library for API calls.
    """
    
    def __init__(
        self,
        endpoint_id: str,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the FastCoref client.
        
        Args:
            endpoint_id: RunPod endpoint ID
            api_key: RunPod API key (defaults to RUNPOD_API_KEY env var)
        """
        # Set API key (runpod library will also check env var)
        if api_key:
            runpod.api_key = api_key
        elif os.environ.get("RUNPOD_API_KEY"):
            runpod.api_key = os.environ["RUNPOD_API_KEY"]
        
        self.endpoint = runpod.Endpoint(endpoint_id)
    
    def _parse_result(self, data: dict) -> CorefResult:
        """Parse a single result from the API response."""
        return CorefResult(
            clusters=data.get("clusters", []),
            clusters_char_offsets=[
                [tuple(offset) for offset in cluster]
                for cluster in data.get("clusters_char_offsets", [])
            ],
            cluster_logits=data.get("cluster_logits"),
        )
    
    def predict(
        self,
        text: str,
        return_logits: bool = False,
        timeout: int = 90
    ) -> CorefResult:
        """
        Run coreference resolution on a single text (synchronous).
        
        Args:
            text: The text to analyze
            return_logits: Whether to return logit scores for cluster pairs
            timeout: Timeout in seconds (default 90)
            
        Returns:
            CorefResult containing detected coreference clusters
        """
        results = self.predict_batch([text], return_logits=return_logits, timeout=timeout)
        return results[0] if results else CorefResult(clusters=[], clusters_char_offsets=[])
    
    def predict_batch(
        self,
        texts: List[str],
        return_logits: bool = False,
        timeout: int = 90
    ) -> List[CorefResult]:
        """
        Run coreference resolution on multiple texts (synchronous).
        
        Args:
            texts: List of texts to analyze
            return_logits: Whether to return logit scores
            timeout: Timeout in seconds (default 90)
            
        Returns:
            List of CorefResult, one per input text
        """
        if not texts:
            return []
        
        payload = {
            "texts": texts,
            "return_logits": return_logits
        }
        
        # run_sync blocks until complete or timeout
        output = self.endpoint.run_sync(payload, timeout=timeout)
        
        # Handle error responses
        if isinstance(output, dict) and "error" in output:
            raise RuntimeError(f"RunPod error: {output['error']}")
        
        # Parse results
        results_list = output.get("results", []) if isinstance(output, dict) else []
        return [self._parse_result(r) for r in results_list]
    
    def predict_async(
        self,
        text: str,
        return_logits: bool = False
    ):
        """
        Run coreference resolution asynchronously (returns immediately).
        
        Args:
            text: The text to analyze
            return_logits: Whether to return logit scores
            
        Returns:
            RunPod Job object - call .status() or .output() to get results
        """
        return self.predict_batch_async([text], return_logits=return_logits)
    
    def predict_batch_async(
        self,
        texts: List[str],
        return_logits: bool = False
    ):
        """
        Run coreference resolution on multiple texts asynchronously.
        
        Args:
            texts: List of texts to analyze
            return_logits: Whether to return logit scores
            
        Returns:
            RunPod Job object - call .status() or .output() to get results
        """
        payload = {
            "texts": texts,
            "return_logits": return_logits
        }
        
        # run() returns immediately with a Job object
        return self.endpoint.run(payload)
    
    def health_check(self) -> bool:
        """Check if the endpoint is healthy."""
        try:
            status = self.endpoint.health()
            return status.get("status") == "healthy" if status else False
        except Exception:
            return False


# Convenience function for simple one-off usage
def resolve_coreferences(
    texts: List[str],
    endpoint_id: str,
    api_key: Optional[str] = None
) -> List[CorefResult]:
    """
    Convenience function for one-off coreference resolution.
    
    Args:
        texts: List of texts to analyze
        endpoint_id: RunPod endpoint ID
        api_key: RunPod API key (optional if RUNPOD_API_KEY is set)
        
    Returns:
        List of CorefResult objects
    """
    client = FastCorefClient(endpoint_id=endpoint_id, api_key=api_key)
    return client.predict_batch(texts)


# Example usage
if __name__ == "__main__":
    import sys
    
    # Check for required environment variables
    endpoint_id = os.environ.get("RUNPOD_FASTCOREF_ENDPOINT_ID")
    if not endpoint_id:
        print("Set RUNPOD_FASTCOREF_ENDPOINT_ID environment variable")
        sys.exit(1)
    
    # Test texts
    texts = [
        "We are so happy to see you using our coref package. This package is very fast!",
        "Barack Obama was president. He served two terms.",
    ]
    
    print("Testing FastCoref client...")
    client = FastCorefClient(endpoint_id=endpoint_id)
    
    for text in texts:
        print(f"\nText: {text}")
        result = client.predict(text)
        print(f"Clusters: {result.clusters}")
        print(f"Offsets: {result.clusters_char_offsets}")
