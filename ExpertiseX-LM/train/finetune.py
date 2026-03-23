import os
import argparse
import torch
from torch.utils.data import random_split
from transformers import AutoTokenizer, AutoConfig, Trainer, TrainingArguments
from dataset import ExpertiseGraphDataset
from model import ExpertiseXLMForSequenceClassification
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    scores = logits.squeeze(-1)
    preds = (scores > 0).astype(int)
    labels_int = labels.astype(int)
    metrics = {"accuracy": accuracy_score(labels_int, preds)}
    if len(np.unique(labels_int)) > 1:
        metrics["auc"] = roc_auc_score(labels_int, scores)
        metrics["f1"] = f1_score(labels_int, preds)
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"d:\C500\Lab306\Reviewer_Recommendation\crawl-data\articles")
    parser.add_argument("--pretrained_model_dir", type=str, default="./pretrain_output/final")
    parser.add_argument("--output_dir", type=str, default="./finetune_output")
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
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
    
    # Train / Val / Test split
    test_size = int(len(full_dataset) * args.test_ratio)
    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    config = AutoConfig.from_pretrained("vinai/phobert-base-v2")
    config.vocab_size = len(tokenizer)
    config.num_labels = 1
    
    # Load model: prefer pretrain checkpoint, fallback to PhoBERT pretrained
    if os.path.exists(os.path.join(args.pretrained_model_dir, "pytorch_model.bin")):
        print(f"Loading MSLM-pretrained model from {args.pretrained_model_dir}...")
        model = ExpertiseXLMForSequenceClassification(config)
        pretrain_state = torch.load(os.path.join(args.pretrained_model_dir, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        # Copy matching keys (encoder + embeddings from pretraining, skip lm_head)
        model_state = model.state_dict()
        loaded = 0
        for key in pretrain_state:
            # Map pretrain keys to finetune keys (both share roberta.*)
            if key in model_state and pretrain_state[key].shape == model_state[key].shape:
                model_state[key] = pretrain_state[key]
                loaded += 1
        model.load_state_dict(model_state)
        print(f"  Loaded {loaded}/{len(model_state)} parameter tensors from pretrain checkpoint")
    else:
        print("No pretrain checkpoint found. Loading directly from PhoBERT pretrained...")
        model = ExpertiseXLMForSequenceClassification.from_pretrained_phobert("vinai/phobert-base-v2", config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        seed=42,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    # Manual save
    save_dir = os.path.join(args.output_dir, "final")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Save test indices for later evaluation
    test_indices = test_dataset.indices
    torch.save(test_indices, os.path.join(args.output_dir, "test_indices.pt"))
    print(f"Saved {len(test_indices)} test indices for evaluation")

if __name__ == "__main__":
    main()
