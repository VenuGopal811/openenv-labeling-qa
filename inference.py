import os
import requests
from openai import OpenAI

# CHECKBOX: Environment variables are present
# CHECKBOX: Defaults are set ONLY for BASE_URL and MODEL_NAME (not HF_TOKEN)
API_BASE_URL = os.getenv("API_BASE_URL", "https://venugopal8115-labelsense-openenv.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o") # or whatever model you're using
HF_TOKEN = os.getenv("HF_TOKEN") # Leave default as None per checklist

# CHECKBOX: All LLM calls use the OpenAI client configured via these variables
client = OpenAI(
    base_url=f"{API_BASE_URL}/v1", 
    api_key=HF_TOKEN
)

def run_inference():
    # CHECKBOX: Stdout logs follow the required structured format (START/STEP/END)
    print("START")
    
    # Your logic here to ping /reset and /step
    # Example:
    # response = requests.post(f"{API_BASE_URL}/reset")
    
    print("STEP: 1")
    # ... more logic ...
    
    print("END")

if __name__ == "__main__":
    run_inference()