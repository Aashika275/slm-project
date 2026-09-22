from pathlib import Path

from tokenizers import Tokenizer


TOKENIZER_FILE = Path(
    "tokenizer/codex_tokenizer_v2.json"
)


tokenizer = Tokenizer.from_file(
    str(TOKENIZER_FILE)
)


text = """
Write a Python function that checks
whether a number is prime.
"""


print("=" * 60)
print("CODEX-SLM TOKENIZER TEST")
print("=" * 60)

print("\nOriginal text:")
print(text)


encoding = tokenizer.encode(text)


print("\nToken IDs:")
print(encoding.ids)


print("\nTokens:")
print(encoding.tokens)


print("\nNumber of tokens:")
print(len(encoding.ids))


decoded = tokenizer.decode(
    encoding.ids
)


print("\nDecoded text:")
print(decoded)


print("\nTokenizer test completed.")
