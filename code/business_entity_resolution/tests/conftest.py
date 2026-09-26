"""Pytest bootstrap for direct package-directory and repository-root test runs."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PACKAGE_ROOT.parents[1]

for path in (PACKAGE_ROOT, CODE_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

# Unit tests must not download large models or depend on local transformer weights.
os.environ.setdefault("IDENTIFAI_SEMANTIC_ENABLED", "0")
os.environ.setdefault("IDENTIFAI_RERANK_TOP_K", "0")
