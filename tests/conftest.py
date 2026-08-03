"""
`data.py` et les pages vivent à la racine du dépôt, pas dans un package installé.
`python -m pytest` insère automatiquement le répertoire courant dans sys.path,
mais l'exécutable `pytest` seul (utilisé par `uv run pytest`, notamment en CI)
ne le fait pas -- on l'ajoute donc explicitement ici.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
