import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType

MODEL_NAME = 'ProsusAI/finbert'
OUTPUT_DIR = 'models/finbert_renewable'

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
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Model saved to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == '__main__':
    train_finbert()