#!/usr/bin/env python3
"""
AI vs Human Text Detection - CLI Tool
Usage: python detect.py "your text here"
       python detect.py --file document.txt
"""

import sys
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# HIGH-END MODEL: RoBERTa-Large LLM Content Detector
# Trained on HC3 and IDMGSP datasets for modern AI detection
# Excellent accuracy with stable downloads (~1GB)
# HIGH-END MODEL: RoBERTa-Large LLM Content Detector
# Trained on HC3 and IDMGSP datasets for modern AI detection
# Excellent accuracy with stable downloads (~1GB)
REMOTE_MODEL_NAME = "SuperAnnotate/roberta-large-llm-content-detector"
LOCAL_MODEL_PATH = "models/roberta_detector"

import os

if os.path.exists(LOCAL_MODEL_PATH):
    MODEL_NAME = LOCAL_MODEL_PATH
    print(f"Using local fine-tuned model: {MODEL_NAME}")
else:
    MODEL_NAME = REMOTE_MODEL_NAME
    print(f"Using remote model: {MODEL_NAME}")

def load_model():
    """Load the pre-trained model and tokenizer"""
    print(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print("✓ Model loaded!\n")
    return tokenizer, model

def analyze_text(text, tokenizer, model):
    """Analyze text and return probabilities"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    human_prob = probs[0][0].item() * 100
    ai_prob = probs[0][1].item() * 100
    
    return human_prob, ai_prob

def display_results(text, human_prob, ai_prob):
    """Display formatted results"""
    print("=" * 60)
    print("TEXT ANALYSIS RESULT")
    print("=" * 60)
    print(f"Text Preview: {text[:100]}{'...' if len(text) > 100 else ''}\n")
    
    print(f"🤖 AI Probability:    {ai_prob:.2f}%")
    print(f"👤 Human Probability: {human_prob:.2f}%\n")
    
    # Verdict
    if ai_prob > 60:
        print("📊 VERDICT: AI GENERATED (High Confidence)")
    elif ai_prob < 40:
        print("📊 VERDICT: HUMAN WRITTEN (High Confidence)")
    else:
        print("📊 VERDICT: UNCERTAIN / MIXED (Low Confidence)")
    
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Detect AI-generated text")
    parser.add_argument("text", nargs="?", help="Text to analyze")
    parser.add_argument("--file", "-f", help="Path to text file")
    
    args = parser.parse_args()
    
    # Get text from arguments or file
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            sys.exit(1)
    elif args.text:
        text = args.text
    else:
        print("❌ Please provide text or use --file option")
        print("Usage: python detect.py \"your text here\"")
        print("       python detect.py --file document.txt")
        sys.exit(1)
    
    if not text.strip():
        print("❌ Error: Empty text provided")
        sys.exit(1)
    
    # Load model and analyze
    tokenizer, model = load_model()
    human_prob, ai_prob = analyze_text(text, tokenizer, model)
    display_results(text, human_prob, ai_prob)

if __name__ == "__main__":
    main()
