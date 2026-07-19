"""Compatibility entrypoint for the standalone submission miner.

This lets the project run with the usual Poker44 command:

    python neurons/miner.py ...

while keeping the actual implementation in the repository root at `miner.py`.
"""

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_MINER = ROOT / "miner.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    runpy.run_path(str(ROOT_MINER), run_name="__main__")
