import os
import argparse
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
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    special_tokens_dict = {'additional_special_tokens': ['[TARGET_PAPER]', '[REVIEWER_PAPER]', '[RESEARCH_AREA]']}
    tokenizer.add_special_tokens(special_tokens_dict)

    eval_csv_path = os.path.join(args.data_dir, "evaluations.csv")
    dataset = ExpertiseGraphDataset(
        data_dir=args.data_dir,
        eval_csv=eval_csv_path,
        tokenizer=tokenizer,
        is_pretrain=False
    )
    
    config = AutoConfig.from_pretrained("vinai/phobert-base-v2")
    config.vocab_size = len(tokenizer)
    config.num_labels = 1 # Regression (MSE) for 1-5 float ratings
    
    # In practice, you'd load the safetensors/pytorch_model.bin from pretraining
    model = ExpertiseXLMForSequenceClassification(config)
    if os.path.exists(args.pretrained_model_dir):
        # We only load state if it's explicitly available, or Trainer can handle `from_pretrained`
        model = ExpertiseXLMForSequenceClassification.from_pretrained(args.pretrained_model_dir, config=config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=8,
        save_steps=200,
        save_total_limit=2,
        logging_steps=10,
        learning_rate=2e-5,
        remove_unused_columns=False,
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))

if __name__ == "__main__":
    main()
