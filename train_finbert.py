import torch
import json
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

MODEL_NAME = 'ProsusAI/finbert'
OUTPUT_DIR = 'models/finbert_renewable'
# FinBERT id2label order, used to label the confusion matrix
LABELS = ['positive', 'negative', 'neutral']

def load_data():
    with open('data/train.json') as f:
        train_data = [json.loads(line) for line in f]
    with open('data/test.json') as f:
        test_data = [json.loads(line) for line in f]
    
    return Dataset.from_dict({
        'text': [x['text'] for x in train_data],
        'label': [int(x['label']) for x in train_data]
    }), Dataset.from_dict({
        'text': [x['text'] for x in test_data],
        'label': [int(x['label']) for x in test_data]
    })

def preprocess_function(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    per_class_f1 = f1_score(labels, predictions, average=None, labels=[0, 1, 2])
    cm = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    print(f"\nConfusion matrix (rows=true, cols=predicted), labels={LABELS}:\n{cm}")
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'f1_positive': per_class_f1[0],
        'f1_negative': per_class_f1[1],
        'f1_neutral': per_class_f1[2],
    }

def train_finbert():
    print("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # num_labels=3 matches FinBERT's native positive/negative/neutral head, so LoRA
    # fine-tunes the pretrained classifier instead of training a random one from scratch.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=['query', 'value'],
        lora_dropout=0.1,
        bias='none',
        task_type=TaskType.SEQ_CLS  # <--- Note: This is fixed for Sequence Classification!
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("Loading datasets...")
    train_dataset, test_dataset = load_data()
    
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True
    )
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    resume_checkpoint = os.environ.get('RESUME_FROM_CHECKPOINT')
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    print(f"Model saved to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == '__main__':
    train_finbert()