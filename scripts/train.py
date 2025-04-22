#!/usr/bin/env python3
import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    default_data_collator
)

def parse_args():
    parser = argparse.ArgumentParser("Fine-tune a chatbot model on review data")
    parser.add_argument(
        "--train_file",
        type=str,
        default=r"C:\Users\susil\amazon-automotive-mlops\data\processed\train.jsonl",
        help="Path to the train JSONL"
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default=r"C:\Users\susil\amazon-automotive-mlops\data\processed\val.jsonl",
        help="Path to the validation JSONL"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=r"C:\Users\susil\amazon-automotive-mlops\data\processed\test.jsonl",
        help="Path to the test JSONL (optional, not used for training)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/flan-t5-small",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\Users\susil\amazon-automotive-mlops\chatbot_model",
        help="Where to save the fine-tuned model"
    )
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    return parser.parse_args()

def preprocess_function(examples, tokenizer, args):
    inputs = [
        "Summarize this review: " + r.get("text", "")
        for r in examples["review"]
    ]
    targets = [r.get("title", "") for r in examples["review"]]
    model_inputs = tokenizer(
        inputs,
        max_length=args.max_input_length,
        truncation=True,
        padding="max_length"
    )
    labels = tokenizer(
        targets,
        max_length=args.max_target_length,
        truncation=True,
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load train/validation as HF datasets
    data_files = {
        "train": args.train_file,
        "validation": args.val_file
    }
    raw_datasets = load_dataset("json", data_files=data_files, field=None)

    # 2) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # 3) Preprocess all splits
    tokenized = raw_datasets.map(
        lambda ex: preprocess_function(ex, tokenizer, args),
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )

    # 4) Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        fp16=torch.cuda.is_available(),
    )

    # 5) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator
    )

    # 6) Train
    trainer.train()

    # 7) Save final model & tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to {args.output_dir}")

if __name__ == "__main__":
    main()
