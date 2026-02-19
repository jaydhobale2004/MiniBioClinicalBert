

import json
from pathlib import Path
from tokenizers import BertWordPieceTokenizer

IN_JSONL = Path("data/corpus/synthea_notes.jsonl")
OUT_DIR = Path("artifacts/tokenizer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TMP_TEXT = Path("data/corpus/all_notes_text.txt")


def main():
    # Convert JSONL into a plain text file (tokenizer training likes plain text)
    with IN_JSONL.open("r", encoding="utf-8") as fin, TMP_TEXT.open("w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            fout.write(obj["text"].replace("\r", "") + "\n")

    # Train WordPiece tokenizer (BERT-style)
    tokenizer = BertWordPieceTokenizer(lowercase=True)
    tokenizer.train(
        files=[str(TMP_TEXT)],
        vocab_size=16000,  # smaller vocab = smaller model; 16k is a good mini choice
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    tokenizer.save_model(str(OUT_DIR))
    print(f"Saved tokenizer files to: {OUT_DIR}")


if __name__ == "__main__":
    main()
