
import json
from pathlib import Path

from tokenizers import Tokenizer


APP_DIR = Path(__file__).resolve().parent

TOKENIZER_PATH = (
    APP_DIR / "tokenizer_assets" / "bpe_tokenizer.json"
)

METADATA_PATH = (
    APP_DIR / "tokenizer_assets" / "tokenizer_metadata.json"
)


PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

VOCAB_SIZE = 8000


def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer metadata not found: {METADATA_PATH}"
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_tokenizer():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found: {TOKENIZER_PATH}"
        )

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    actual_size = tokenizer.get_vocab_size()

    if actual_size != VOCAB_SIZE:
        raise ValueError(
            f"Expected vocabulary size {VOCAB_SIZE}, "
            f"but found {actual_size}."
        )

    return tokenizer


def encode_text(text, add_special_tokens=True):
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    tokenizer = load_tokenizer()

    return tokenizer.encode(
        text,
        add_special_tokens=add_special_tokens
    ).ids


def decode_tokens(token_ids, skip_special_tokens=True):
    if not isinstance(token_ids, (list, tuple)):
        raise TypeError("token_ids must be a list or tuple.")

    tokenizer = load_tokenizer()

    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=skip_special_tokens
    )


# Load once when the module is imported.
tokenizer = load_tokenizer()
metadata = load_metadata()
