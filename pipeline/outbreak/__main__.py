"""`python pipeline/outbreak <slug>` (from the repository root) or `python -m outbreak <slug>` (from pipeline/)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # pipeline/, for common and prepare_city

from outbreak.adapter import main  # noqa: E402

main()
