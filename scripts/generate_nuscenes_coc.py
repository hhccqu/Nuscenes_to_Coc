#!/home/hhc/anaconda3/envs/drivestudio/bin/python
"""CLI wrapper for nuScenes CoC generation."""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nuscenes_coc.cli import main


if __name__ == "__main__":
    main()
