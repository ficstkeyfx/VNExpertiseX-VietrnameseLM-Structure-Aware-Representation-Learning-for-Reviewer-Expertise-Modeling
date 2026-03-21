import os
import json
import random
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

class NodeType:
    TARGET_PAPER = 0
    REVIEWER_PAPER = 1
    RESEARCH_AREA = 2

class ExpertiseGraphDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        eval_csv: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 256,
        is_pretrain: bool = True
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_pretrain = is_pretrain
        
        self.articles_dir = data_dir
        
        self.research_areas = {} # paper_id -> [areas]
        self.author_to_papers = {} # author_name -> [paper_ids]
        self.all_paper_ids = []
        
        self._load_articles()
            
        self.pairs = []
        if eval_csv and os.path.exists(eval_csv):
            df = pd.read_csv(eval_csv, delimiter='\t')
            for index, row in df.iterrows():
                reviewer_id = str(row['ParticipantID'])
                for i in range(1, 11):
                    paper_col = f'Paper{i}'
                    exp_col = f'Expertise{i}'
                    if paper_col in row and pd.notna(row[paper_col]):
                        paper_id = str(row[paper_col])
                        if exp_col in row and pd.notna(row[exp_col]):
                            score = float(row[exp_col])
                        else:
                            score = 1.0 # Default positive match
                        self.pairs.append((reviewer_id, paper_id, score))
            
            # Autogenerate negative samples (0.0) if only positive labels exist
            if not self.is_pretrain:
                all_scores = set(score for _, _, score in self.pairs)
                if len(all_scores) == 1 and 1.0 in all_scores:
                    self._generate_negative_samples()
        else:
            # Proxy author-paper pairs
            self._build_pairs_from_participants()
            if not self.is_pretrain:
                self._generate_negative_samples()
            
    def _load_articles(self):
        import glob
        from tqdm import tqdm
        article_files = glob.glob(os.path.join(self.articles_dir, "*.json"))
        
        print(f"Discovered {len(article_files)} articles total. Loading a sample of 2000 for efficiency...")
        article_files = article_files[:2000]
        
        for p_file in tqdm(article_files, desc="Parsing internal relationships"):
            paper_id = os.path.basename(p_file).replace(".json", "")
            self.all_paper_ids.append(paper_id)
            
            try:
                with open(p_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 1. Map Research Area
                    rf = data.get("research_field", "")
                    if rf: self.research_areas[paper_id] = [rf]
                    
                    # 2. Map Authors
                    authors_str = data.get("author", "")
                    if authors_str:
                        authors = [a.strip() for a in authors_str.split(",") if a.strip()]
                        for author in authors:
                            if author not in self.author_to_papers:
                                self.author_to_papers[author] = []
                            self.author_to_papers[author].append(paper_id)
            except Exception as e:
                pass
                
    def _build_pretrain_pairs(self):
        self._build_pairs_from_participants()

    def _build_pairs_from_participants(self):
        # Build Positive pairs using the mapped authors
        for author_id, papers in self.author_to_papers.items():
            for p_id in papers[:10]: # Limit to 10 articles max per author
                self.pairs.append((author_id, p_id, 1.0))

    def _generate_negative_samples(self):
        # Generate random negative matches for 1:1 ratio
        num_positives = len(self.pairs)
        if not self.all_paper_ids: return
        
        negative_pairs = []
        for i in range(num_positives):
            rev_id = self.pairs[i][0]
            true_paper = self.pairs[i][1]
            rand_paper = random.choice(self.all_paper_ids)
            while rand_paper == true_paper or (rev_id in self.author_to_papers and rand_paper in self.author_to_papers[rev_id]):
                rand_paper = random.choice(self.all_paper_ids)
            negative_pairs.append((rev_id, rand_paper, 0.0))
        
        self.pairs.extend(negative_pairs)
        random.shuffle(self.pairs)
        
    def _read_paper(self, paper_id):
        path = os.path.join(self.articles_dir, f"{paper_id}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def _read_participant(self, reviewer_id):
        # Mock participant document from the index
        if reviewer_id in self.author_to_papers:
            return {"papers": [{"paperId": pid} for pid in self.author_to_papers[reviewer_id]]}
        return None

    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        reviewer_id, target_paper_id, label = self.pairs[idx]
        
        # 1. Target paper
        target_paper = self._read_paper(target_paper_id)
        if not target_paper:
            target_paper = {"title": "Unknown Title", "abstract": ""}
            
        # 2. Reviewer profile
        participant = self._read_participant(reviewer_id)
        reviewer_papers = []
        if participant and "papers" in participant:
            for p in participant["papers"][:5]: # Take top 5 papers as budget
                p_data = self._read_paper(p["paperId"])
                if p_data:
                    reviewer_papers.append(p_data)
                    
        # 3. Research Areas for the target
        target_areas = self.research_areas.get(target_paper_id, [])

        return self._serialize_subgraph(target_paper, reviewer_papers, target_areas, label)
        
    def _serialize_subgraph(self, target_paper, reviewer_papers, target_areas, label):
        """
        Serialize node: [NodeType][ResearchArea][Title][Abstract]
        Builds the expertise position matrix.
        Pi = [E0, E1, E2, E3, E4, E5]
        """
        input_ids = []
        expertise_positions = []
        node_masks = [] # To group tokens for nodes in MSLM
        
        # Helper to add a tokenized string block with position tracking
        def add_node(text_block, e0, e1, e2, e3, e4, node_idx):
            tokens = self.tokenizer.encode(text_block, add_special_tokens=False)
            for e5, token in enumerate(tokens):
                if len(input_ids) >= self.max_seq_length - 2: # reserve for CLS, SEP
                    break
                input_ids.append(token)
                e5_clamped = min(e5, 255)
                expertise_positions.append([e0, e1, e2, e3, e4, e5_clamped])
                node_masks.append(node_idx)
        
        # Start constructing
        # E0 Role: Target=0, RevPaper=1, Area=2
        # E1 Distance: Target=0, otherwise=1
        # E2 Reviewer Authored: Target=0, RevPaper=1
        # E3 Area depth (dummy=0)
        # E4 Node Type: Paper=0, Area=1
        
        node_counter = 0
        
        # TARGET PAPER
        target_text = f"[TARGET_PAPER] {target_paper.get('title', '')} {target_paper.get('abstract', '')}"
        add_node(target_text, e0=0, e1=0, e2=0, e3=0, e4=0, node_idx=node_counter)
        node_counter += 1
        
        # RESEARCH AREAS
        if target_areas:
            area_text = f"[RESEARCH_AREA] {' '.join(target_areas)}"
            add_node(area_text, e0=2, e1=1, e2=0, e3=1, e4=1, node_idx=node_counter)
            node_counter += 1
            
        # REVIEWER PAPERS
        for rp in reviewer_papers:
            rp_text = f"[REVIEWER_PAPER] {rp.get('title', '')} {rp.get('abstract', '')}"
            add_node(rp_text, e0=1, e1=1, e2=1, e3=0, e4=0, node_idx=node_counter)
            node_counter += 1

        # Add CLS and SEP
        final_input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        final_positions = [[0]*6] + expertise_positions + [[0]*6]
        final_node_masks = [-1] + node_masks + [-1] # -1 for special tokens
        
        # Pad to max length
        pad_len = self.max_seq_length - len(final_input_ids)
        attention_mask = [1] * len(final_input_ids) + [0] * pad_len
        final_input_ids += [self.tokenizer.pad_token_id] * pad_len
        final_positions += [[0]*6] * pad_len
        final_node_masks += [-1] * pad_len

        return {
            "input_ids": torch.tensor(final_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "expertise_positions": torch.tensor(final_positions, dtype=torch.long),
            "node_masks": torch.tensor(final_node_masks, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.float)
        }

class MSLM_DataCollator:
    def __init__(self, tokenizer, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        
    def __call__(self, features):
        batch = {}
        for key in features[0].keys():
            if key != "labels":
                batch[key] = torch.stack([f[key] for f in features])
                
        # MSLM Node-level masking
        input_ids = batch["input_ids"].clone()
        labels = input_ids.clone()
        
        for i in range(len(features)):
            node_masks = batch["node_masks"][i]
            valid_nodes = torch.unique(node_masks[node_masks != -1]).tolist()
            if not valid_nodes:
                continue
                
            num_tokens = (batch["attention_mask"][i] == 1).sum().item() - 2 # ignore cls, sep
            mask_budget = int(num_tokens * self.mask_prob)
            
            random.shuffle(valid_nodes)
            masked_count = 0
            
            for node_idx in valid_nodes:
                if masked_count >= mask_budget:
                    break
                    
                node_positions = (node_masks == node_idx).nonzero(as_tuple=True)[0]
                
                for pos in node_positions:
                    prob = random.random()
                    if prob < 0.8:
                        input_ids[i, pos] = self.tokenizer.mask_token_id
                    elif prob < 0.9:
                        input_ids[i, pos] = random.randint(0, self.tokenizer.vocab_size - 1)
                    # 10% keep original
                    
                masked_count += len(node_positions)
                
            # Ignore loss for non-masked tokens by setting label to -100
            labels[i][input_ids[i] == batch["input_ids"][i]] = -100
            
        batch["input_ids"] = input_ids
        batch["mlm_labels"] = labels
        batch["classification_labels"] = torch.stack([f["labels"] for f in features])
        
        return batch
