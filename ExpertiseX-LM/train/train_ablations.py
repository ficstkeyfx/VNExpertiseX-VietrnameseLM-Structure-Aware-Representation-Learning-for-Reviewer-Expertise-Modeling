"""
Training script for Ablation Study  (Section 3.5)

Trains separate models for each of the 5 configurations:

  1. full           MSLM pretrain → finetune  (all features)
  2. no_expertise   MSLM pretrain → finetune  (expertise positions zeroed)
  3. no_pretrain    finetune from PhoBERT directly  (no MSLM step)
  4. no_area        MSLM pretrain → finetune  (research-area nodes removed)
  5. no_negative    MSLM pretrain → finetune  (positive-only, no neg sampling)

Config 5 reuses the "full" pretrain checkpoint (MSLM is unsupervised and
unaffected by negative sampling, which only applies at the finetune stage).

Usage
-----
    # Train one configuration
    python train_ablations.py --ablation full
    python train_ablations.py --ablation no_expertise
    python train_ablations.py --ablation no_pretrain
    python train_ablations.py --ablation no_area
    python train_ablations.py --ablation no_negative

    # Train ALL five at once (takes a while)
    python train_ablations.py --ablation all

    # Quick test run (5 gradient steps per stage)
    python train_ablations.py --ablation all --pretrain_max_steps 5 --finetune_max_steps 5

Output
------
    ablation_models/
    ├── full/
    │   ├── pretrain_output/final/   (MSLM checkpoint)
    │   └── finetune_output/final/   (classification checkpoint)
    ├── no_expertise/
    │   ├── pretrain_output/final/
    │   └── finetune_output/final/
    ├── no_pretrain/
    │   └── finetune_output/final/
    ├── no_area/
    │   ├── pretrain_output/final/
    │   └── finetune_output/final/
    └── no_negative/
        └── finetune_output/final/   (reuses full pretrain)
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, AutoConfig, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from dataset import ExpertiseGraphDataset, MSLM_DataCollator
from model import ExpertiseXLMForPreTraining, ExpertiseXLMForSequenceClassification


# ═══════════════════ ablation registry ══════════════════════

ABLATION_CONFIGS = {
    "full": dict(
        do_pretrain=True,
        use_expertise=True,  use_areas=True,  use_negatives=True,
    ),
    "no_expertise": dict(
        do_pretrain=True,
        use_expertise=False, use_areas=True,  use_negatives=True,
    ),
    "no_pretrain": dict(
        do_pretrain=False,
        use_expertise=True,  use_areas=True,  use_negatives=True,
    ),
    "no_area": dict(
        do_pretrain=True,
        use_expertise=True,  use_areas=False, use_negatives=True,
    ),
    "no_negative": dict(
        do_pretrain=True,
        use_expertise=True,  use_areas=True,  use_negatives=False,
    ),
}


# ═══════════════════ dataset wrapper ════════════════════════

class ZeroExpertiseDataset(Dataset):
    """Wraps any dataset to zero out expertise_positions (ablation: standard position only)."""

    def __init__(self, base_dataset):
        self.base = base_dataset
        self.pairs = getattr(base_dataset, "pairs", [])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        item["expertise_positions"] = torch.zeros_like(item["expertise_positions"])
        return item


# ═══════════════════ shared helpers ═════════════════════════

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "[TARGET_PAPER]", "[REVIEWER_PAPER]", "[RESEARCH_AREA]"
        ]
    })
    return tokenizer


def get_config(tokenizer, num_labels=None):
    config = AutoConfig.from_pretrained("vinai/phobert-base-v2")
    config.vocab_size = len(tokenizer)
    if num_labels is not None:
        config.num_labels = num_labels
    return config


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


def _free_model(model):
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════ pretrain stage ═════════════════════════

def run_pretrain(
    data_dir, output_dir, tokenizer, config, *,
    use_expertise=True, use_areas=True,
    num_epochs=5, max_steps=-1,
):
    """
    MSLM pre-training.

    Ablation toggles
    ----------------
    use_expertise=False  →  expertise_positions zeroed (standard position only)
    use_areas=False      →  research-area nodes omitted from serialization
    """
    banner = (f"PRETRAIN  expertise={use_expertise}  areas={use_areas}  "
              f"epochs={num_epochs}  max_steps={max_steps}")
    print(f"\n{'─'*60}\n {banner}\n Output: {output_dir}\n{'─'*60}")

    dataset = ExpertiseGraphDataset(
        data_dir=data_dir, eval_csv=None,
        tokenizer=tokenizer, is_pretrain=True,
        use_areas=use_areas,
    )
    if not use_expertise:
        dataset = ZeroExpertiseDataset(dataset)

    collator = MSLM_DataCollator(tokenizer=tokenizer, mask_prob=0.15)
    model = ExpertiseXLMForPreTraining.from_pretrained_phobert(
        "vinai/phobert-base-v2", config,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
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

    Trainer(
        model=model, args=training_args,
        train_dataset=dataset, data_collator=collator,
    ).train()

    save_dir = os.path.join(output_dir, "final")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"  ✓ Pretrain saved → {save_dir}")

    _free_model(model)
    return save_dir


# ═══════════════════ finetune stage ═════════════════════════

def run_finetune(
    data_dir, pretrain_dir, output_dir, tokenizer, config, *,
    use_expertise=True, use_areas=True, use_negatives=True,
    from_phobert=False,
    num_epochs=5, max_steps=-1,
):
    """
    Classification fine-tuning.

    Ablation toggles
    ----------------
    use_expertise=False  →  expertise_positions zeroed
    use_areas=False      →  area nodes omitted
    use_negatives=False  →  positive-only training (no neg sampling)
    from_phobert=True    →  init from PhoBERT base (skip MSLM pretrain)
    """
    src = "PhoBERT base" if from_phobert else pretrain_dir
    banner = (f"FINETUNE  expertise={use_expertise}  areas={use_areas}  "
              f"negatives={use_negatives}\n Source: {src}")
    print(f"\n{'─'*60}\n {banner}\n Output: {output_dir}\n{'─'*60}")

    config.num_labels = 1

    # ── dataset ──
    full_dataset = ExpertiseGraphDataset(
        data_dir=data_dir, eval_csv=None,
        tokenizer=tokenizer, is_pretrain=False,
        use_areas=use_areas,
        use_negatives=use_negatives,
    )
    if not use_expertise:
        full_dataset = ZeroExpertiseDataset(full_dataset)

    test_size  = int(len(full_dataset) * 0.1)
    val_size   = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size - test_size
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  Split → Train {train_size} / Val {val_size} / Test {test_size}")

    # ── model ──
    if from_phobert:
        model = ExpertiseXLMForSequenceClassification.from_pretrained_phobert(
            "vinai/phobert-base-v2", config,
        )
    else:
        model_bin = os.path.join(pretrain_dir, "pytorch_model.bin")
        if os.path.exists(model_bin):
            model = ExpertiseXLMForSequenceClassification(config)
            pt_state = torch.load(model_bin, map_location="cpu", weights_only=True)
            m_state  = model.state_dict()
            loaded = 0
            for k in pt_state:
                if k in m_state and pt_state[k].shape == m_state[k].shape:
                    m_state[k] = pt_state[k]
                    loaded += 1
            model.load_state_dict(m_state)
            print(f"  Loaded {loaded}/{len(m_state)} tensors from pretrain")
        else:
            print(f"  Warning: {model_bin} missing → falling back to PhoBERT base")
            model = ExpertiseXLMForSequenceClassification.from_pretrained_phobert(
                "vinai/phobert-base-v2", config,
            )

    # ── training ──
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
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

    Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    ).train()

    # ── save ──
    save_dir = os.path.join(output_dir, "final")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    torch.save(test_ds.indices, os.path.join(output_dir, "test_indices.pt"))
    print(f"  ✓ Finetune saved → {save_dir}")
    print(f"  ✓ Test indices ({len(test_ds.indices)}) saved")

    _free_model(model)
    return save_dir


# ═══════════════ single-ablation orchestrator ═══════════════

def run_ablation(
    ablation_name, data_dir, output_base_dir, *,
    pretrain_epochs=5,  finetune_epochs=5,
    pretrain_max_steps=-1, finetune_max_steps=-1,
    reuse_pretrain_dir=None,
):
    """Run one ablation config end-to-end (pretrain → finetune)."""
    cfg = ABLATION_CONFIGS[ablation_name]

    print(f"\n{'═'*60}")
    print(f" ABLATION CONFIG: {ablation_name}")
    print(f"   do_pretrain   = {cfg['do_pretrain']}")
    print(f"   use_expertise = {cfg['use_expertise']}")
    print(f"   use_areas     = {cfg['use_areas']}")
    print(f"   use_negatives = {cfg['use_negatives']}")
    print(f"{'═'*60}")

    tokenizer = get_tokenizer()
    base_config = get_config(tokenizer)

    abl_dir = os.path.join(output_base_dir, ablation_name)
    os.makedirs(abl_dir, exist_ok=True)

    pretrain_dir = None

    # ── pretrain ──
    if cfg["do_pretrain"]:
        if reuse_pretrain_dir and os.path.exists(
            os.path.join(reuse_pretrain_dir, "pytorch_model.bin")
        ):
            print(f"  Reusing pretrain checkpoint: {reuse_pretrain_dir}")
            pretrain_dir = reuse_pretrain_dir
        else:
            pretrain_out = os.path.join(abl_dir, "pretrain_output")
            pretrain_dir = run_pretrain(
                data_dir, pretrain_out, tokenizer, base_config,
                use_expertise=cfg["use_expertise"],
                use_areas=cfg["use_areas"],
                num_epochs=pretrain_epochs,
                max_steps=pretrain_max_steps,
            )

    # ── finetune ──
    ft_config = get_config(tokenizer, num_labels=1)
    finetune_out = os.path.join(abl_dir, "finetune_output")
    finetune_dir = run_finetune(
        data_dir, pretrain_dir, finetune_out, tokenizer, ft_config,
        use_expertise=cfg["use_expertise"],
        use_areas=cfg["use_areas"],
        use_negatives=cfg["use_negatives"],
        from_phobert=(not cfg["do_pretrain"]),
        num_epochs=finetune_epochs,
        max_steps=finetune_max_steps,
    )

    print(f"\n  ✓ Ablation '{ablation_name}' complete → {finetune_dir}")
    return finetune_dir


# ═══════════════════════ CLI ════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train models for Ablation Study (Section 3.5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ablation", type=str, required=True,
        choices=list(ABLATION_CONFIGS.keys()) + ["all"],
        help="Which ablation to train (or 'all' for everything)",
    )
    parser.add_argument("--data_dir", type=str,
                        default=r"d:\C500\Lab306\Reviewer_Recommendation\crawl-data\articles")
    parser.add_argument("--output_base_dir", type=str, default="./ablation_models")
    parser.add_argument("--pretrain_epochs",    type=int, default=5)
    parser.add_argument("--finetune_epochs",    type=int, default=5)
    parser.add_argument("--pretrain_max_steps", type=int, default=-1,
                        help="Override pretrain epoch count (-1 = use epochs)")
    parser.add_argument("--finetune_max_steps", type=int, default=-1,
                        help="Override finetune epoch count (-1 = use epochs)")
    args = parser.parse_args()

    common = dict(
        data_dir=args.data_dir,
        output_base_dir=args.output_base_dir,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        pretrain_max_steps=args.pretrain_max_steps,
        finetune_max_steps=args.finetune_max_steps,
    )

    if args.ablation == "all":
        # ── 1. full (pretrain + finetune) ──
        run_ablation("full", **common)
        full_pt = os.path.join(
            args.output_base_dir, "full", "pretrain_output", "final"
        )

        # ── 2. no_expertise (own pretrain — different embedding regime) ──
        run_ablation("no_expertise", **common)

        # ── 3. no_pretrain (finetune only, from PhoBERT) ──
        run_ablation("no_pretrain", **common)

        # ── 4. no_area (own pretrain — different input graph) ──
        run_ablation("no_area", **common)

        # ── 5. no_negative (reuse full pretrain — only finetune differs) ──
        run_ablation("no_negative", **common, reuse_pretrain_dir=full_pt)

        print(f"\n{'═'*60}")
        print(f" ALL 5 ABLATION MODELS TRAINED")
        print(f" Models saved under: {args.output_base_dir}/")
        print(f"{'═'*60}")
        print("\nNext step — run evaluation:")
        print(f"  cd ../eval")
        print(f"  python eval_ablation.py \\")
        print(f"    --model_dir         ..\\train\\ablation_models\\full\\finetune_output\\final \\")
        print(f"    --no_pretrain_model_dir ..\\train\\ablation_models\\no_pretrain\\finetune_output\\final \\")
        print(f"    --no_negative_model_dir ..\\train\\ablation_models\\no_negative\\finetune_output\\final")

    else:
        reuse = None
        if args.ablation == "no_negative":
            candidate = os.path.join(
                args.output_base_dir, "full", "pretrain_output", "final",
                "pytorch_model.bin",
            )
            if os.path.exists(candidate):
                reuse = os.path.dirname(candidate)
                print(f"  Will reuse full pretrain: {reuse}")

        run_ablation(args.ablation, **common, reuse_pretrain_dir=reuse)


if __name__ == "__main__":
    main()
