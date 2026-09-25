"""Repository-root convenience entrypoint for the packaged submission code."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

from business_entity_resolution.run import main


if __name__ == "__main__":
    main()
