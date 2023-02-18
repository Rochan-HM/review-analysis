from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL = "deep-learning-analytics/automatic-title-generation"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)


def main(text: str) -> str:
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=10, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
