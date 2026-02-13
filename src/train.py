import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
import os
import numpy as np

# Configuration
MODEL_NAME = "roberta-base" # Using base model for speed/memory efficiency
OUTPUT_DIR = "models/roberta_detector_hq" # Save to a new HQ folder
MAX_LENGTH = 512
NUM_EPOCHS = 3
BATCH_SIZE = 4 # Safer batch size for local training
MAX_SAMPLES = 2000 # Enough to prove concept, too much = slow

print(f"Preparing to train {MODEL_NAME} for High Quality Bot Detection...")
print(f"Saving to: {OUTPUT_DIR}")

# 1. Load Dataset
print("Loading dataset...")
# PRIORITY: use local CSV (High Quality generated data)
if os.path.exists("data/train.csv"):
    print("Found local data/train.csv. Using it for training.")
    df = pd.read_csv("data/train.csv")
    dataset = Dataset.from_pandas(df)
else:
    print("Downloading/Loading HC3 Dataset...")
    try:
        # enable trust_remote_code to handle the HC3 loading script
        dataset = load_dataset("Hello-SimpleAI/HC3", name="all", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset directly: {e}")
        raise RuntimeError("Could not load dataset. Please ensure data/train.csv exists.")

# 2. Preprocess Data
print("Processing data...")
texts = []
labels = []

# If using local CSV, it's already formatted as text/label
if "text" in dataset.features:
    print("Using standard text/label format from CSV...")
    for item in dataset:
        texts.append(str(item['text']))
        labels.append(int(item['label']))
else:
    # HC3 processing fallback
    count = 0
    for item in dataset:
        if count >= MAX_SAMPLES:
            break
            
        human_answers = item.get('human_answers', [])
        ai_answers = item.get('chatgpt_answers', [])
        
        # Add Human examples (Label 0)
        for answer in human_answers:
            if len(answer.strip()) > 50: 
                texts.append(answer)
                labels.append(0)
                count += 1
                
        # Add AI examples (Label 1)
        for answer in ai_answers:
            if len(answer.strip()) > 50:
                texts.append(answer)
                labels.append(1)
                count += 1

# Create DataFrame
df = pd.DataFrame({'text': texts, 'label': labels})
# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# Trim to exact max samples (since we added in loop)
df = df.head(MAX_SAMPLES)

print(f"Total training examples: {len(df)}")
print(f"Human samples: {len(df[df['label']==0])}")
print(f"AI samples: {len(df[df['label']==1])}")

# Split
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Convert back to HF Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# 3. Tokenization
print(f"Tokenizing with {MODEL_NAME}...")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 4. Model Setup
print("Loading model...")
model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "Human", 1: "AI"},
    label2id={"Human": 0, "AI": 1},
    ignore_mismatched_sizes=True
)

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    logging_steps=10,
    save_total_limit=1,
    remove_unused_columns=False,
    use_cpu=not torch.cuda.is_available() # Updated from no_cuda
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # processing_class is new, tokenizer is old. 
    # Let's rely on model's config or pass via processing_class if supported, but safer to omit if collator handles it.
    # Actually, we can just set data_collator
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

# 7. Start Training
print("Starting training... (This may take 15-30 minutes)")
trainer.train()

# 8. Save Final Model
print(f"Saving model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✓ Training complete! New High-Quality model is ready.")
