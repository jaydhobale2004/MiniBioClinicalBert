"""
Fine-tunes Bio_ClinicalBERT as a SimCSE bi-encoder on synthetic clinical notes.

Usage:
    python artifacts/finetune_sbert.py
    python artifacts/finetune_sbert.py --force_retrain
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import Dataset
from torch import nn
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, set_seed


DEFAULT_CHECKPOINT = "emilyalsentzer/Bio_ClinicalBERT"


class BertSentenceTransformer(nn.Module):
    """BERT bi-encoder: mean pooling -> optional projection -> L2-normalize."""

    def __init__(self, model_name_or_path: str, projection_dim: int = 384, dropout_p: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        hidden_size  = self.encoder.config.hidden_size

        self.projection = (
            nn.Sequential(nn.Linear(hidden_size, projection_dim), nn.LayerNorm(projection_dim))
            if projection_dim < hidden_size else None
        )
        self.dropout = nn.Dropout(dropout_p)

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self) -> bool:
        return getattr(self.encoder, "is_gradient_checkpointing", False)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden = out.last_hidden_state
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        if self.training:
            pooled = self.dropout(pooled)
        if self.projection is not None:
            pooled = self.projection(pooled)

        return torch.nn.functional.normalize(pooled, p=2, dim=1)


def load_notes(jsonl_path: Path):
    with jsonl_path.open("r", encoding="utf-8") as f:
        return [json.loads(l)["text"] for l in f]


def compute_metrics(_) -> Dict[str, Any]:
    return {}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("CUDA not available — using CPU (training will be slow).")
    return torch.device("cpu")


def model_already_exists(output_dir: str) -> bool:
    out = Path(output_dir)
    return out.exists() and any(
        f.suffix in (".safetensors", ".bin") for f in out.iterdir()
    )


def main(
    mlm_checkpoint: str           = DEFAULT_CHECKPOINT,
    output_dir: str               = "artifacts/sbert_bi_encoder",
    num_train_epochs: int         = 5,
    per_device_train_batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    learning_rate: float          = 2e-5,
    warmup_ratio: float           = 0.1,
    max_len: Optional[int]        = 256,
    projection_dim: int           = 384,
    temperature: float            = 0.05,
    seed: int                     = 42,
    force_retrain: bool           = False,
):
    if not force_retrain and model_already_exists(output_dir):
        print(f"Fine-tuned model already exists at '{output_dir}'.")
        print("Pass --force_retrain to overwrite it.")
        return

    set_seed(seed)
    device = get_device()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Base model: {mlm_checkpoint}")

    in_jsonl = Path("data/corpus/synthea_notes.jsonl")
    if not in_jsonl.exists():
        raise FileNotFoundError(f"Missing {in_jsonl}. Run make_notes_from_csv.py first.")

    texts   = load_notes(in_jsonl)
    print(f"Loaded {len(texts)} notes.")

    dataset  = Dataset.from_dict({"text": texts}).train_test_split(test_size=0.1, seed=seed)

    tokenizer     = AutoTokenizer.from_pretrained(mlm_checkpoint)
    model         = BertSentenceTransformer(mlm_checkpoint, projection_dim, dropout_p=0.1).to(device)
    model_max_len = model.encoder.config.max_position_embeddings
    eff_max_len   = model_max_len if max_len is None else min(max_len, model_max_len)
    print(f"max_len={eff_max_len}  (model limit={model_max_len})")

    def tokenize_fn(batch):
        enc = tokenizer(
            batch["text"], truncation=True,
            max_length=eff_max_len, padding="max_length", return_tensors="pt",
        )
        return {k: v.tolist() for k, v in enc.items()}

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    class SBERTTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.can_return_loss = True

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            ids   = inputs["input_ids"]
            mask  = inputs["attention_mask"]
            ttype = inputs["token_type_ids"] if "token_type_ids" in inputs else None

            # same note run twice with different dropout masks gives us a positive pair (SimCSE)
            z1 = model(input_ids=ids, attention_mask=mask, token_type_ids=ttype)
            z2 = model(input_ids=ids, attention_mask=mask, token_type_ids=ttype)

            logits_12 = torch.mm(z1, z2.t()) / temperature
            logits_21 = torch.mm(z2, z1.t()) / temperature
            labels    = torch.arange(z1.size(0), device=z1.device)
            loss_fn   = nn.CrossEntropyLoss()
            loss      = 0.5 * (loss_fn(logits_12, labels) + loss_fn(logits_21, labels))
            return (loss, logits_12) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        logging_steps=50,
        eval_steps=200,
        eval_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=(device.type == "cuda"),
        dataloader_num_workers=0,
        report_to="none",
    )

    trainer = SBERTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # save only the inner encoder so AutoModel.from_pretrained works at inference time
    model.encoder.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nSaved fine-tuned encoder to: {output_dir}")
    print("Next step: python build_index.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mlm_checkpoint",              default=DEFAULT_CHECKPOINT)
    p.add_argument("--output_dir",                  default="artifacts/sbert_bi_encoder")
    p.add_argument("--num_train_epochs",  type=int,   default=5)
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate",     type=float, default=2e-5)
    p.add_argument("--warmup_ratio",      type=float, default=0.1)
    p.add_argument("--max_len",           type=int,   default=256)
    p.add_argument("--projection_dim",    type=int,   default=384)
    p.add_argument("--temperature",       type=float, default=0.05)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--force_retrain",     action="store_true",
                   help="Overwrite existing model and retrain from scratch.")
    args = p.parse_args()
    main(**vars(args))
