import requests
import os

def embed(text: str):
    response = requests.post(
        "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
        headers={"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"},
        json={"inputs": text, "options": {"wait_for_model": True}}
    )
    return response.json()