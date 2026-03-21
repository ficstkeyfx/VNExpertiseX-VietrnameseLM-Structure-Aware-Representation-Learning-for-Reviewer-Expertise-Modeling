import os
import argparse
from transformers import AutoTokenizer, AutoConfig, Trainer, TrainingArguments
from dataset import ExpertiseGraphDataset, MSLM_DataCollator
from model import ExpertiseXLMForPreTraining

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"d:\C500\Lab306\Reviewer_Recommendation\goldstandard-reviewer-paper-match\data")
    parser.add_argument("--output_dir", type=str, default="./pretrain_output")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", local_files_only=True)
    # Add special tokens
    special_tokens_dict = {'additional_special_tokens': ['[TARGET_PAPER]', '[REVIEWER_PAPER]', '[RESEARCH_AREA]']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # Load dataset for pretraining (we can just pass eval_csv=None, dataset will need a basic participant list)
    # For demonstration we reuse evaluations.csv as a source of valid (Reviewer, Paper) pairs, 
    # but in pure MSLM this should be unlabeled sampled pairings from all papers/reviewers
    eval_csv_path = os.path.join(args.data_dir, "evaluations.csv")
    dataset = ExpertiseGraphDataset(
        data_dir=args.data_dir,
        eval_csv=eval_csv_path, # Using this merely to grab test pairs easily
        tokenizer=tokenizer,
        is_pretrain=True
    )
    
    collator = MSLM_DataCollator(tokenizer=tokenizer, mask_prob=0.15)

    config = AutoConfig.from_pretrained("vinai/phobert-base-v2", local_files_only=True)
    # match vocab size to tokenizer after adding special tokens
    config.vocab_size = len(tokenizer)

    model = ExpertiseXLMForPreTraining(config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        max_steps=10,
        per_device_train_batch_size=4,
        save_steps=10,
        save_total_limit=1,
        logging_steps=5,
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
