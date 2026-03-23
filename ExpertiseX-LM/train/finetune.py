import os
import argparse
import torch
from torch.utils.data import random_split
from transformers import AutoTokenizer, AutoConfig, Trainer, TrainingArguments
from dataset import ExpertiseGraphDataset
from model import ExpertiseXLMForSequenceClassification

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"d:\C500\Lab306\Reviewer_Recommendation\crawl-data\articles")
    parser.add_argument("--pretrained_model_dir", type=str, default="./pretrain_output/final")
    parser.add_argument("--output_dir", type=str, default="./finetune_output")
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    special_tokens_dict = {'additional_special_tokens': ['[TARGET_PAPER]', '[REVIEWER_PAPER]', '[RESEARCH_AREA]']}
    tokenizer.add_special_tokens(special_tokens_dict)

    full_dataset = ExpertiseGraphDataset(
        data_dir=args.data_dir,
        eval_csv=None,
        tokenizer=tokenizer,
        is_pretrain=False
    )
    
    # Train/Val split
    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    config = AutoConfig.from_pretrained("vinai/phobert-base-v2")
    config.vocab_size = len(tokenizer)
    config.num_labels = 1
    
    model = ExpertiseXLMForSequenceClassification(config)
    if os.path.exists(args.pretrained_model_dir):
        model = ExpertiseXLMForSequenceClassification.from_pretrained(args.pretrained_model_dir, config=config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    
    # Manual save to bypass safetensors shared-weights issue
    save_dir = os.path.join(args.output_dir, "final")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    main()

