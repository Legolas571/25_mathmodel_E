from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent

SRC_DIR = ROOT / "数据集" / "源域数据集"
TGT_DIR = ROOT / "数据集" / "目标域数据集"
OUT_DIR = BASE_DIR / "outputs"
FIG_DIR = BASE_DIR / "figures"
CACHE_DIR = OUT_DIR / "cache"

for _d in (OUT_DIR, FIG_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FS_12K = 12000
FS_48K = 48000
FS_TGT = 32000

RPM_TGT_PRIOR = 600.0
RPM_SEARCH_REL = 0.15
DEFAULT_RPM = 1797.0

BEARINGS = {
    "SKF6205": dict(N=9, d=0.3126, D=1.537),
    "SKF6203": dict(N=9, d=0.2656, D=1.122),
}

GROUPS = {
    "12k_DE": dict(fs=FS_12K, bearing="SKF6205"),
    "12k_FE": dict(fs=FS_12K, bearing="SKF6203"),
    "48k_DE": dict(fs=FS_48K, bearing="SKF6205"),
    "48k_Normal": dict(fs=FS_48K, bearing="SKF6205"),
    "target": dict(fs=FS_TGT, bearing=None),
}

CLASSES = ["B", "IR", "OR", "N"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
OR_POSITION = {3: "Orthogonal", 6: "Centered", 12: "Opposite"}

WIN_SEC_SIGNAL = 0.128
WIN_SEC_FEATURE = 0.5
OVERLAP_TRAIN = 0.5
OVERLAP_TEST = 0.0

ORDER_POINTS_PER_REV = 256
N_ORDERS = 32
N_TOP_PEAKS = 8
PEAK_TOL_REL = 0.03
N_FOLDS = 5
SEED = 42

DEFAULT_BAND = (2000.0, 5000.0)
MIN_BAND_WIDTH = 200.0
