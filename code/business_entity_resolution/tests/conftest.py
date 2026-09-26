import os

# Unit tests must be deterministic and must not download large ML models. The
# production CLI keeps semantic retrieval enabled by default.
os.environ.setdefault("IDENTIFAI_SEMANTIC_ENABLED", "0")
