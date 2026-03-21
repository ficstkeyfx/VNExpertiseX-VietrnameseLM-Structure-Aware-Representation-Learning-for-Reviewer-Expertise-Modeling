import os
import argparse
import glob
import json
import pandas as pd
import torch
import random
from transformers import AutoTokenizer, AutoConfig
import sys

# To import from train folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'train'))
from dataset import ExpertiseGraphDataset
from model import ExpertiseXLMForSequenceClassification

def get_top_k(scores, k=3):
    return scores.sort_values(ascending=False).head(k)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"d:\C500\Lab306\Reviewer_Recommendation\crawl-data\articles")
    parser.add_argument("--model_dir", type=str, default=r"..\train\finetune_output\final")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--num_test_papers", type=int, default=2)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", local_files_only=True)
    special_tokens_dict = {'additional_special_tokens': ['[TARGET_PAPER]', '[REVIEWER_PAPER]', '[RESEARCH_AREA]']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # Khởi tạo instance dataset chỉ để tái sử dụng module tiền xử lý (serialization)
    dataset = ExpertiseGraphDataset(
        data_dir=args.data_dir,
        eval_csv="dummy.csv", # Ghi đè file eval để không tự load hết các cặp
        tokenizer=tokenizer,
        is_pretrain=False
    )
    dataset.pairs = [] # Xoá cache pair
    
    config = AutoConfig.from_pretrained("vinai/phobert-base-v2", local_files_only=True)
    config.vocab_size = len(tokenizer)
    config.num_labels = 1
    
    model = ExpertiseXLMForSequenceClassification(config)
    if os.path.exists(args.model_dir):
        print(f"Loading model from {args.model_dir}...")
        try:
            model = ExpertiseXLMForSequenceClassification.from_pretrained(args.model_dir, config=config)
        except Exception as e:
            pass
    else:
        print("Pre-trained fine-tuning model not found, using random weights for demonstration.")

    model.eval()
    
    # 1. Quét danh sách tất cả các ứng viên tiềm năng từ index
    candidate_reviewers = []
    for author, papers in dataset.author_to_papers.items():
        candidate_reviewers.append({"id": author, "authored_papers": papers})
    
    print(f"Loaded {len(candidate_reviewers)} candidate reviewers.")
    candidate_reviewers = candidate_reviewers[:50] # Demo: Giới hạn top 50 ứng viên để xếp hạng cho lẹ
    
    # 2. Ngẫu nhiên bốc ra một vài bài báo mục tiêu đang cần tìm Reviewer
    test_paper_ids = random.sample(dataset.all_paper_ids, min(args.num_test_papers, len(dataset.all_paper_ids)))
    
    print("\n--- TOP-K Reviewer Recommendations ---")
    with torch.no_grad():
        for target_paper_id in test_paper_ids:
            print(f"\nTarget Paper: {target_paper_id}")
            
            predictions = []
            for reviewer in candidate_reviewers:
                reviewer_id = reviewer["id"]
                
                # --- CHỐNG TRÙNG LẶP (CONFLICT OF INTEREST) ---
                # Loại bỏ người phản biện nếu họ chính là một trong các tác giả của bài báo này!
                if target_paper_id in reviewer["authored_papers"]:
                    # print(f" - Bỏ qua Reviewer {reviewer_id} vì là tác giả của bài báo.")
                    continue
                
                # Bắt đầu tính điểm 
                target_paper = dataset._read_paper(target_paper_id)
                if not target_paper: continue
                
                participant_data = dataset._read_participant(reviewer_id)
                reviewer_papers = []
                if participant_data and "papers" in participant_data:
                    for p in participant_data["papers"][:5]:
                        p_data = dataset._read_paper(p["paperId"])
                        if p_data: reviewer_papers.append(p_data)
                        
                target_areas = dataset.research_areas.get(target_paper_id, [])
                
                features = dataset._serialize_subgraph(target_paper, reviewer_papers, target_areas, label=0.0)
                input_ids = features['input_ids'].unsqueeze(0)
                attention_mask = features['attention_mask'].unsqueeze(0)
                expertise_positions = features['expertise_positions'].unsqueeze(0)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    expertise_positions=expertise_positions
                )
                
                predicted_score = outputs.logits.item()
                predictions.append({
                    "reviewer_id": reviewer_id,
                    "predicted_score": predicted_score
                })
            
            df_preds = pd.DataFrame(predictions)
            if not df_preds.empty:
                top_k = get_top_k(df_preds.set_index('reviewer_id')['predicted_score'], k=args.top_k)
                for rank, (rev_id, score) in enumerate(top_k.items(), 1):
                    # Vì model dùng Sigmoid Logits để phân tích đúng sai, score càng cao tỷ lệ khớp càng lớn.
                    print(f" {rank}. Reviewer {rev_id} - Predicted Match Score: {score:.4f}")
            else:
                print(" No eligible reviewers found after filtering authors.")

if __name__ == "__main__":
    main()
