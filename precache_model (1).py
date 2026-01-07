## This script will run when we first create the Docker container
## It will pre-cache the classification model to speed up first use
## HuggingFace models are cached in the ~/.cache/huggingface directory by default
## When we invoke the model later, it will look for it in the cache first

from transformers import pipeline
import sys

## Classificationnmodel
classification_model_name = "NaveenTNS/mental-roberta"

try:
    # Try loading the model to cache it
    pipeline("text-classification", model=classification_model_name)
except Exception as e:
    print(f"Error pre-caching model: {e}")
    sys.exit(1)