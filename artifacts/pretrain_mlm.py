import json
from pathlib import Path

from datasets import Dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ----------------------------
# Inputs/Outputs
# ----------------------------
IN_JSONL = Path("data/corpus/synthea_notes.jsonl")
TOKENIZER_DIR = "artifacts/tokenizer"
OUT_MODEL_DIR = "artifacts/mini_biobert_mlm"

MAX_LEN = 256  # start with 128 if your machine is slow; 256 gives richer context


def load_texts():
    texts = []
    with IN_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
    return texts


def main():
    # 1) Load note texts
    texts = load_texts()
    ds = Dataset.from_dict({"text": texts})

    # split into train/test so we can monitor learning
    ds = ds.train_test_split(test_size=0.02, seed=42)

    # 2) Load tokenizer we trained
    tok = BertTokenizerFast.from_pretrained(TOKENIZER_DIR)

    # 3) Tokenize the dataset
    def tokenize_batch(batch):
        return tok(
            batch["text"],
            truncation=True,
            max_length=MAX_LEN,
            padding=False,  # padding will be done later by the collator
        )

    tokenized = ds.map(tokenize_batch, batched=True, remove_columns=["text"])

    # 4) Data collator does the MLM masking for us:
    #    It chooses 15% tokens and replaces them with [MASK]/random/original.
    collator = DataCollatorForLanguageModeling(
        tokenizer=tok,
        mlm=True,
        mlm_probability=0.15
    )

    # 5) Define a "mini" BERT configuration (encoder-only)
    config = BertConfig(
        vocab_size=tok.vocab_size,

        hidden_size=256,           # embedding size
        num_hidden_layers=4,       # number of Transformer encoder layers
        num_attention_heads=4,     # attention heads
        intermediate_size=1024,    # feed-forward size

        hidden_act="gelu",
        max_position_embeddings=MAX_LEN + 2,
        type_vocab_size=2,
    )

    model = BertForMaskedLM(config)

    # 6) Training settings
    # If you have no GPU, set fp16=False and lower batch_size
    args = TrainingArguments(
        output_dir=OUT_MODEL_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.06,

        num_train_epochs=3,

        logging_steps=50,
        eval_steps=500,
        eval_strategy="steps",

        save_steps=500,
        save_total_limit=2,

        fp16=True,        # set False if CPU-only or fp16 errors
        report_to="none",
    )

    # 7) Trainer runs the training loop
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=collator,
    )

    trainer.train()

    # 8) Save final model + tokenizer
    trainer.save_model(OUT_MODEL_DIR)
    tok.save_pretrained(OUT_MODEL_DIR)
    print(f"Saved MiniBioBERT MLM model to: {OUT_MODEL_DIR}")


if __name__ == "__main__":
    main()
