import os
import requests

from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

load_dotenv()

MODEL = "deep-learning-analytics/automatic-title-generation"
API_URL = "https://api-inference.huggingface.co/models/deep-learning-analytics/automatic-title-generation"
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)


def _generate(text: str) -> str:
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=10, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main(texts: List[str]) -> str:
    print("Using local model")
    return [_generate(text) for text in texts]


def api(texts: List[str]) -> str:
    try:
        print("Using API")
        headers = {"Authorization": f"Bearer {API_KEY}"}

        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "inputs": texts,
                "parameters": {
                    "max_length": 10,
                    "num_beams": 4,
                    "early_stopping": True,
                },
            },
        )
        print(response)

        return [resp["generated_text"] for resp in response.json()]
    except Exception as e:
        print(e)
        return main(texts)
