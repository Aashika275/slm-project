from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import BPEDecoder
from tokenizers.trainers import BpeTrainer


# ============================================================
# PATHS
# ============================================================

CORPUS_FILE = Path(
    "data/processed/tokenizer_corpus.txt"
)

TOKENIZER_DIR = Path(
    "tokenizer"
)

TOKENIZER_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TOKENIZER_FILE = (
    TOKENIZER_DIR /
    "codex_tokenizer_v2.json"
)


# ============================================================
# CREATE TOKENIZER
# ============================================================

tokenizer = Tokenizer(
    BPE(
        unk_token="<|unk|>"
    )
)


# ============================================================
# PRE-TOKENIZER
# ============================================================

tokenizer.pre_tokenizer = Whitespace()


# ============================================================
# DECODER
# ============================================================

tokenizer.decoder = BPEDecoder(
    suffix="</w>"
)


# ============================================================
# SPECIAL TOKENS
# ============================================================

special_tokens = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]


# ============================================================
# TRAINER
# ============================================================

trainer = BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    special_tokens=special_tokens,
    end_of_word_suffix="</w>",
)


# ============================================================
# TRAIN
# ============================================================

print("=" * 60)
print("CODEX-SLM TOKENIZER V2 TRAINING")
print("=" * 60)

print()
print("Corpus:")
print(CORPUS_FILE)

if not CORPUS_FILE.exists():

    raise FileNotFoundError(
        f"Tokenizer corpus not found:\n"
        f"{CORPUS_FILE}"
    )

print()
print("Training tokenizer...")

tokenizer.train(
    [str(CORPUS_FILE)],
    trainer
)


# ============================================================
# SAVE
# ============================================================

tokenizer.save(
    str(TOKENIZER_FILE)
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("TOKENIZER V2 TRAINING COMPLETE")
print("=" * 60)

print()
print("Tokenizer saved to:")
print(TOKENIZER_FILE)

print()
print("Vocabulary size:")
print(tokenizer.get_vocab_size())

print()
print("Tokenizer training completed successfully.")
