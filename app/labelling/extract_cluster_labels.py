import os
import requests
import streamlit as st

from collections import OrderedDict
from typing import List
from stqdm import stqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

load_dotenv()

MODEL = "deep-learning-analytics/automatic-title-generation"
API_URL = "https://api-inference.huggingface.co/models/deep-learning-analytics/automatic-title-generation"
API_KEY = os.getenv("HUGGINGFACE_API_KEY") or st.secrets["HUGGINGFACE_API_KEY"]

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)


def clean_sentence(text):
    unique_phrases = list(OrderedDict.fromkeys(text.split()))
    cleaned_str = " ".join(unique_phrases).title()
    return cleaned_str


def _generate(text: str) -> str:
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=10, num_beams=4, early_stopping=True)
    sent = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_sentence(sent)


def main(texts: List[str]) -> List[str]:
    print("Using local model")
    res = []
    for i in stqdm(range(len(texts))):
        res.append(_generate(texts[i]))

    return res


def api(texts: List[str]) -> List[str]:
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

        return [clean_sentence(resp["generated_text"]) for resp in response.json()]
    except Exception as e:
        print(e)
        return main(texts)
