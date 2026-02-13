from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import numpy as np

model_path = "models/roberta_detector"

try:
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
except Exception as e:
    print(f"Model not found at {model_path}. Please train the model first.")
    exit(1)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.detach().numpy()

import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = "Artificial intelligence is transforming industries and reshaping the way we work."
    print(f"Analyzing text: '{text}'")
    
    probs = predict(text)
    ai_probability = probs[0][1]
    ai_score = ai_probability * 100
    
    print(f"\n🔍 Analysis Result:")
    print(f"-------------------")
    print(f"AI Detection Score: {ai_score:.2f}%")
    print(f"Label: {'🤖 AI Generated' if ai_score > 50 else '👤 Human Written'}")
