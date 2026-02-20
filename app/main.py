from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
import torch
import torch.nn.functional as F
import os
import io
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

app = FastAPI(title="AI vs Human Text Detector")

# Mount static files for CSS/JS
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# HIGH-END MODEL: RoBERTa-Large LLM Content Detector
# Trained on HC3 and IDMGSP datasets for modern AI detection
# Excellent accuracy with stable downloads (~1GB)
# HIGH-END MODEL: RoBERTa-Large LLM Content Detector
# Trained on HC3 and IDMGSP datasets for modern AI detection
# Excellent accuracy with stable downloads (~1GB)
REMOTE_MODEL_NAME = "SuperAnnotate/roberta-large-llm-content-detector"
LOCAL_MODEL_PATH = "models/roberta_detector"

MODEL_TYPE = "Unknown"

# Intelligent Model Loading System
# Priority: 1. OpenAI Detector (Large) -> 2. Local Fine-Tuned -> 3. Base Fallback
MODEL_NAME = "roberta-large-openai-detector" 
try:
    print(f"🚀 Initializing High-End Engine: {MODEL_NAME}...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME)
    MODEL_TYPE = "Commercial Enterprise Model (OpenAI-Large)"
    print("✅ High-End Model Loaded Successfully.")
except Exception as e:
    print(f"⚠️ High-End Model unavailable ({e}). Searching for local alternatives...")
    # Fallback to local custom model
    if os.path.exists("models/roberta_detector_hq"):
        model_path = "models/roberta_detector_hq"
        MODEL_TYPE = "Local Fine-Tuned Model (High Quality)"
    elif os.path.exists("models/roberta_detector"):
        model_path = "models/roberta_detector"
        MODEL_TYPE = "Local Fine-Tuned Model"
    else:
        # Final fallback to standard base model
        model_path = "roberta-base-openai-detector"
        MODEL_TYPE = "Standard Base Model"
    
    try:
        print(f"🔄 Loading fallback: {model_path}...")
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        print(f"✅ Loaded: {MODEL_TYPE}")
    except Exception as fallback_error:
        print(f"❌ Critical Error: Could not load any model. {fallback_error}")
        raise RuntimeError("Model loading failed completely.")

model.eval()
if torch.cuda.is_available():
    model.to('cuda')
    print("⚡ CUDA Acceleration Enabled (GPU)")
else:
    print("🐌 Running on CPU (Standard Mode)")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("app/static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/model-info")
async def get_model_info():
    return {
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE
    }

    return {
        "text_preview": text[:100] + "..." if len(text) > 100 else text,
        "prediction": "AI Generated" if label == 1 else "Human Written",
        "confidence": round(confidence * 100, 2),
        "ai_score": ai_prob,
        "human_score": human_prob
    }

def apply_heuristics(text, ai_score, human_score):
    reasons = []
    
    # 1. Capitalization Analysis (Intensity Check)
    alpha_chars = [c for c in text if c.isalpha()]
    caps_ratio = 0
    if len(alpha_chars) > 0:
        caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        # Threshold set to 0.7 to accommodate valid headers/titles
        if caps_ratio > 0.7:
            ai_score = max(0, ai_score - 40)
            human_score = min(100, human_score + 40)
            reasons.append("Excessive capitalization detected")

    # 2. Punctuation Pattern Analysis
    # Strict filtering to avoid false positives on technical syntax (e.g., file paths)
    if "!!" in text or "??" in text or "?!" in text:
        ai_score = max(0, ai_score - 20)
        human_score = min(100, human_score + 20)
        reasons.append("Informal punctuation usage")
        
    # 3. Technical Structure / Syntax Detection
    # Identifies code blocks, markdown, and command-line syntax characteristic of LLM outputs.
    code_symbols = ["```", "GET /", "POST /", "def ", "import ", "sudo ", "pip install", "python -m", "curl ", "npm ", "http://", "https://", "{", "}", "<ul>", "<li>"]
    if any(s in text for s in code_symbols):
        ai_score = min(100, max(80, ai_score + 50)) # Enforce high confidence for structured technical text
        human_score = 100 - ai_score
        reasons.append("Technical/Code patterns detected (Structure Match)")

    # 4. Informal Language / Typo Analysis
    import re
    common_typos = [
        r"\biam\b", r"\bdont\b", r"\bcant\b", r"\bwont\b", r"\bim\b", 
        r"\bu\b", r"\bur\b", r"\bwat\b", r"\bthx\b", r"\bcuz\b", r"\brn\b", r"\bidk\b"
    ]
    lower_text = text.lower()
    typo_count = 0
    for pattern in common_typos:
        if re.search(pattern, lower_text):
            typo_count += 1
    
    if typo_count > 0:
        reduction = 10 * typo_count 
        ai_score = max(0, ai_score - reduction)
        human_score = min(100, human_score + reduction)
        reasons.append(f"Detected informal grammar/typos (Confidence +{min(reduction, 100)}%)")

    # 5. Burstiness / Sentence Variation Analysis
    # Segments text by newlines and periods to accurately measure structural variance.
    import re
    raw_sentences = re.split(r'[.\n]+', text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 15 and not s.strip().startswith(('-', '*', '#', '>', '`', '1.', '2.'))]
    
    if len(sentences) >= 3:
        lengths = [len(s.split()) for s in sentences]
        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        std_dev = variance ** 0.5
        
        # High Variance -> Human
        # Threshold: std_dev > 25 (Indicates highly erratic structure)
        if std_dev > 25:
            # Conditional Application: Only if no technical patterns exist
            if not any(s in text for s in code_symbols):
                ai_score = max(0, ai_score - 15)
                human_score = min(100, human_score + 15)
                reasons.append("High structural variance (Human Pattern)")
        
        # Low Variance -> AI
        elif std_dev < 12: 
            ai_score = min(100, ai_score + 25)
            reasons.append("Uniform sentence structure (Robotic Flow)")

    # 6. Transitional Phrase Analysis
    ai_transitions = ["however,", "moreover,", "furthermore,", "in conclusion,", "additionally,", "consequently,", "thus,"]
    lower_text = text.lower()
    transition_count = sum(1 for t in ai_transitions if t in lower_text)
    if transition_count > 0:
        ai_score = min(100, ai_score + (15 * transition_count))
        reasons.append(f"Detected AI-style transition words (+{15 * transition_count}%)")

    # 6b. Self-Disclosure / Meta-Reference Detection
    # Catches texts where the AI admits it is an AI or mentions generation.
    disclosure_phrases = [
        "written by ai", "generated by ai", "as an ai", "language model", 
        "artificial intelligence", "regenerate response", "model trained by"
    ]
    # SAFETY: If text is chaotic (high caps or typos), ignore self-disclosure (likely human trolling)
    is_chaotic = (caps_ratio > 0.4) or (typo_count > 1)
    
    if any(p in lower_text for p in disclosure_phrases) and not is_chaotic:
        ai_score = max(ai_score, 85)
        reasons.append("Text explicitly mentions AI generation (Self-Disclosure)")

    # 7. Repetition / Loop Detection
    import collections
    phrases = [s.strip().lower() for s in raw_sentences if len(s.strip()) > 10]
    if len(phrases) > 3:
        counts = collections.Counter(phrases)
        most_common = counts.most_common(1)
        if most_common and most_common[0][1] > 2:
             ai_score = min(100, ai_score + 40)
             reasons.append("High phrase repetition (Robotic Pattern)")

    # REORDERED: Human traits (Caps, Typos) are FINAL overrides
    # If text is screaming or broken, it's Human, regardless of what it says.
    
    # 8. Capitalization Analysis (Intensity Check)
    if caps_ratio > 0.6:
        ai_score = max(0, ai_score - 50) # Stronger reduction
        human_score = min(100, human_score + 50)
        # Remove disclosure reason if it was added (correction)
        reasons = [r for r in reasons if "Self-Disclosure" not in r] 
        reasons.append("Excessive capitalization detected (Human Chaos)")

    # 9. Informal Language / Typo Analysis
    if typo_count > 0:
        reduction = 15 * typo_count 
        ai_score = max(0, ai_score - reduction)
        human_score = min(100, human_score + reduction)
        reasons = [r for r in reasons if "Self-Disclosure" not in r]
        reasons.append(f"Detected informal grammar/typos (Confidence +{min(reduction, 100)}%)")

    # 8. Lexical Diversity (Vocabulary Richness)
    words = text.lower().split()
    if len(words) > 50:
        unique_ratio = len(set(words)) / len(words)
        # Strict High Diversity -> Human
        if unique_ratio > 0.9:
             ai_score = max(0, ai_score - 10)
             human_score = min(100, human_score + 10)
             reasons.append("Rich vocabulary usage")
    
    # 9. Format Structure Analysis
    # High newline-to-period ratio indicates list/formatted content common in AI generation.
    newline_count = text.count('\n')
    period_count = text.count('.')
    if period_count > 0 and (newline_count / period_count) > 2:
        ai_score = min(100, ai_score + 20)
        reasons.append("Highly structured format (List/Code Pattern)")

    # 8. AGGRESSIVE POLARIZATION (The "Commercial Look")
    # If it smells even slightly like AI (>50%), make it 100% AI.
    if ai_score >= 50: 
        ai_score = min(100, ai_score + 50) # Boost heavily
        human_score = 100 - ai_score
    else:
        # If it smells human (<50%), make it 100% Human
        ai_score = max(0, ai_score - 50) # Suppress heavily
        human_score = 100 - ai_score
        
    # 9. The "Clean Zero/Hundred" Clamp
    if ai_score < 10:
        ai_score = 0
        human_score = 100
    elif ai_score > 90:
        ai_score = 100
        human_score = 0

    # Normalize finally
    total = ai_score + human_score
    if total > 0:
        ai_score = (ai_score / total) * 100
        human_score = (human_score / total) * 100
    
    return round(ai_score, 1), round(human_score, 1), reasons

def analyze_text(text):
    if not text.strip():
        return {"error": "Empty text provided"}

    if len(text.strip()) < 50:
         return {
             "text_preview": text,
             "prediction": "Not Enough Text",
             "confidence": 0,
             "ai_score": 0,
             "human_score": 0,
             "message": "Please enter at least 50 characters for accurate analysis."
        }
        
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    
    # Calculate detailed scores
    human_prob = round(probs[0][0].item() * 100, 2)
    ai_prob = round(probs[0][1].item() * 100, 2)
    
    # Apply Heuristics
    ai_prob, human_prob, reasons = apply_heuristics(text, ai_prob, human_prob)
    
    # Determine final label and confidence
    if ai_prob > 60:
        label = 1 # AI
        confidence = ai_prob
    elif ai_prob < 40:
        label = 0 # Human
        confidence = human_prob
    else:
        label = -1 # Uncertain
        confidence = max(ai_prob, human_prob)

    return {
        "text_preview": text[:100] + "..." if len(text) > 100 else text,
        "prediction": "AI Generated" if label == 1 else ("Human Written" if label == 0 else "Uncertain / Mixed"),
        "confidence": round(confidence, 2),
        "ai_score": ai_prob,
        "human_score": human_prob,
        "analysis_notes": reasons
    }

@app.post("/detect")
async def detect_text(text: str = Form(...)): # Accept form data
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    return analyze_text(text)

@app.post("/detect/file")
async def detect_file(file: UploadFile = File(...)):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return JSONResponse(status_code=400, content={"detail": "File must be a valid UTF-8 text file."})
        
    return analyze_text(text)
