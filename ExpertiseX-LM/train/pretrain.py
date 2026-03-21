import os
import argparse
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
    # match vocab size to tokenizer after adding special tokens
    config.vocab_size = len(tokenizer)

    model = ExpertiseXLMForPreTraining(config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        learning_rate=5e-5,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))

if __name__ == "__main__":
    main()
