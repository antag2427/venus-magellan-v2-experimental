from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Standalone defaults must be supplied by the upgrade pack itself.
import src.paths  # noqa: F401,E402
