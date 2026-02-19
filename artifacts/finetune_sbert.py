"""
finetune_sbert.py

Fine-tune an SBERT bi-encoder on top of your MLM checkpoint using SimCSE-style contrastive learning.
This improves retrieval embeddings by learning that semantically similar notes are close.

Usage:
    python artifacts/finetune_sbert.py --mlm_checkpoint auto \
        --output_dir artifacts/sbert_bi_encoder --num_train_epochs 3
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from datasets import Dataset
from torch import nn
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, set_seed

DEFAULT_LOCAL_CHECKPOINT = "artifacts/mini_biobert_mlm"
DEFAULT_REMOTE_FALLBACKS = [
    "emilyalsentzer/Bio_ClinicalBERT",
    "dmis-lab/biobert-base-cased-v1.1",
    "bert-base-uncased",
]


class BertSentenceTransformer(nn.Module):
    """Simple BERT-based bi-encoder: BERT -> mean pooling -> optional projection -> normalize."""

    def __init__(self, model_name_or_path: str, projection_dim: int = 384, dropout_p: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = self.encoder.config.hidden_size

        if projection_dim != hidden_size:
            self.projection = nn.Linear(hidden_size, projection_dim)
        else:
            self.projection = None

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        if self.projection is not None:
            pooled = self.projection(pooled)

        if self.training:
            pooled = self.dropout(pooled)

        return torch.nn.functional.normalize(pooled, p=2, dim=1)


def load_notes(jsonl_path: Path):
    texts = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts


def compute_metrics(_) -> Dict[str, Any]:
    return {}


def resolve_checkpoint(checkpoint_arg: str) -> str:
    """
    Resolve model checkpoint for fast-path training:
    - explicit value: use as-is
    - auto: prefer local MLM checkpoint if present, else use first remote fallback
    """
    if checkpoint_arg != "auto":
        return checkpoint_arg

    if Path(DEFAULT_LOCAL_CHECKPOINT).exists():
        print(f"Using local checkpoint: {DEFAULT_LOCAL_CHECKPOINT}")
        return DEFAULT_LOCAL_CHECKPOINT

    fallback = DEFAULT_REMOTE_FALLBACKS[0]
    print(f"Local checkpoint not found. Falling back to remote checkpoint: {fallback}")
    return fallback


def main(
    mlm_checkpoint: str = "auto",
    output_dir: str = "artifacts/sbert_bi_encoder",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    max_len: int | None = None,
    projection_dim: int = 384,
    seed: int = 42,
):
    set_seed(seed)
    resolved_checkpoint = resolve_checkpoint(mlm_checkpoint)

    in_jsonl = Path("data/corpus/synthea_notes.jsonl")
    texts = load_notes(in_jsonl)
    print(f"Loaded {len(texts)} texts for SBERT fine-tuning.")

    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.train_test_split(test_size=0.1, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(resolved_checkpoint)
    model = BertSentenceTransformer(resolved_checkpoint, projection_dim=projection_dim, dropout_p=0.1)
    model_max_len = int(model.encoder.config.max_position_embeddings)
    effective_max_len = model_max_len if max_len is None else min(max_len, model_max_len)
    print(f"Tokenization max_len={effective_max_len} (model limit={model_max_len})")

    def tokenize_fn(batch):
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=effective_max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.tolist() for k, v in encoded.items()}

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    class SBERTTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Two independent forward passes create the positive pair via dropout noise.
            z1 = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids"),
            )
            z2 = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids"),
            )

            temperature = 0.05
            logits_12 = torch.mm(z1, z2.t()) / temperature
            logits_21 = torch.mm(z2, z1.t()) / temperature

            labels = torch.arange(z1.size(0), device=z1.device)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = 0.5 * (loss_fn(logits_12, labels) + loss_fn(logits_21, labels))
            return (loss, {"embeddings_view1": z1, "embeddings_view2": z2}) if return_outputs else loss

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=50,
        eval_steps=500,
        eval_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none",
    )

    trainer = SBERTTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved SBERT bi-encoder to: {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune SBERT bi-encoder on clinical notes")
    p.add_argument(
        "--mlm_checkpoint",
        type=str,
        default="auto",
        help=(
            "Base checkpoint path/name. Use 'auto' to prefer local artifacts/mini_biobert_mlm "
            "and fallback to emilyalsentzer/Bio_ClinicalBERT."
        ),
    )
    p.add_argument("--output_dir", type=str, default="artifacts/sbert_bi_encoder")
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--projection_dim", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    main(
        mlm_checkpoint=args.mlm_checkpoint,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_len=args.max_len,
        projection_dim=args.projection_dim,
        seed=args.seed,
    )

