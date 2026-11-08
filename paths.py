"""Central path resolution for the released code.

Every path can be overridden with an environment variable, so the same code
runs from the repository checkout or from a data directory elsewhere.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = Path(os.environ.get("DSTAR_DATA", REPO / "data"))
OUT = Path(os.environ.get("DSTAR_OUT", REPO / "outputs"))

PLANT_CSV = Path(os.environ.get("DSTAR_PLANT_CSV", DATA / "plant" / "interval_data.csv"))
GPVISC_CSV = Path(os.environ.get("DSTAR_GPVISC_CSV", DATA / "gpvisc" / "dataset+sciglass_lp.csv"))

OUT.mkdir(parents=True, exist_ok=True)
