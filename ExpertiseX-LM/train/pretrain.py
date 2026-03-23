import os
import argparse
import torch
from transformers import AutoTokenizer, AutoConfig, Trainer, TrainingArguments
from dataset import ExpertiseGraphDataset, MSLM_DataCollator
from model import ExpertiseXLMForPreTraining

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"d:\C500\Lab306\Reviewer_Recommendation\crawl-data\articles")
    parser.add_argument("--output_dir", type=str, default="./pretrain_output")
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=-1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    # Add special tokens
    special_tokens_dict = {'additional_special_tokens': ['[TARGET_PAPER]', '[REVIEWER_PAPER]', '[RESEARCH_AREA]']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # Load dataset for pretraining
    dataset = ExpertiseGraphDataset(
        data_dir=args.data_dir,
        eval_csv=None, 

        tokenizer=tokenizer,
        is_pretrain=True
    )
    
    collator = MSLM_DataCollator(tokenizer=tokenizer, mask_prob=0.15)

    config = AutoConfig.from_pretrained("vinai/phobert-base-v2")
    config.vocab_size = len(tokenizer)

    # Load PhoBERT pretrained weights properly
    model = ExpertiseXLMForPreTraining.from_pretrained_phobert("vinai/phobert-base-v2", config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        seed=42,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
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
