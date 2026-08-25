"""Retrieval-augmented generation over backend/assets/data/phill-context-en.md.

There is no external embedding model in this stack (Groq is only used for
chat completion), so `get_embedding` builds a small deterministic vector
with a signed feature-hashing trick: each token is hashed into one of
EMBEDDING_DIM buckets with a +1/-1 sign, and the resulting vector is
L2-normalized. Cosine similarity between two such vectors then reduces to a
plain dot product, which is what `retrieve` uses to rank context chunks.
"""

import hashlib
import re
from pathlib import Path

import numpy as np

from . import config

CONTEXT_PATH = Path(__file__).resolve().parents[1] / "assets" / "data" / "phill-context-en.md"

EMBEDDING_DIM = 256

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def get_embedding(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float64)
    tokens = _TOKEN_RE.findall(text.lower())

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


class _KnowledgeBase:
    """Lazily-loaded, cached split of the context file into chunks."""

    def __init__(self) -> None:
        self.preamble: str = ""
        self.instructions: str = ""
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None

    def load(self) -> None:
        if self.embeddings is not None:
            return

        text = CONTEXT_PATH.read_text(encoding="utf-8")
        parts = re.split(r"\n(?=## )", text)

        self.preamble = parts[0].strip()
        knowledge_chunks: list[str] = []
        instructions = ""

        for part in parts[1:]:
            part = part.strip()
            if part.startswith("## 8."):
                instructions = part
            else:
                knowledge_chunks.append(part)

        self.instructions = instructions
        self.chunks = knowledge_chunks
        self.embeddings = np.stack([get_embedding(chunk) for chunk in knowledge_chunks])


_kb = _KnowledgeBase()


def retrieve(query: str, top_k: int = config.TOP_K_CHUNKS) -> list[str]:
    _kb.load()
    if not _kb.chunks:
        return []

    query_vector = get_embedding(query)
    scores = _kb.embeddings @ query_vector
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [_kb.chunks[i] for i in top_indices]


def build_system_prompt(query: str) -> str:
    _kb.load()
    context_block = "\n\n".join(retrieve(query))

    return (
        f"{_kb.preamble}\n\n"
        "## Retrieved context\n\n"
        f"{context_block}\n\n"
        f"{_kb.instructions}"
    )
