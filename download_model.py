#!/usr/bin/env python3
"""
Download and cache the LingMessCoref model during Docker build.
Patches transformers to use eager attention for Longformer compatibility.
"""

import os

# Force eager attention implementation before importing transformers
os.environ['TRANSFORMERS_ATTENTION_IMPLEMENTATION'] = 'eager'

# Monkey-patch transformers to use eager attention by default
import transformers.modeling_utils as mu

@classmethod
def patched_autoset(cls, config, *args, **kwargs):
    config._attn_implementation = 'eager'
    return config

mu.PreTrainedModel._autoset_attn_implementation = patched_autoset

# Now import and load the model
from fastcoref import LingMessCoref

print('Downloading LingMessCoref model...')
model = LingMessCoref(device='cpu')
print('Model downloaded successfully')
