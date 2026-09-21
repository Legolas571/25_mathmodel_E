import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent
TASK1_DIR = ROOT / "data_anls_and_feature_extraction"
TASK2_DIR = ROOT / "source_domain_fault_diagnosis"
T1_OUT = TASK1_DIR / "outputs"

if str(TASK2_DIR) not in sys.path:
    # append, NOT insert: task 2 also has config.py / data_io.py, and inserting
    # its directory in front would shadow this task's own modules.
    sys.path.append(str(TASK2_DIR))

# mirrors of task-2 config values needed by the imported FeaturePipeline
SCALER = "robust"
WINSORIZE = None

SRC_FEATURES = T1_OUT / "features_source.csv"
TGT_FEATURES = T1_OUT / "features_target.csv"
SPLITS_ALL = TASK2_DIR / "results" / "splits_all.json"
SOURCE_MODEL_PKL = TASK2_DIR / "models" / "source_model_F2.pkl"
TARGET_RPM = T1_OUT / "target_rpm.csv"

RES_DIR = BASE_DIR / "results"
FIG_DIR = BASE_DIR / "figures"
LABEL_DIR = BASE_DIR / "labels"
MODEL_DIR = BASE_DIR / "models"
for _d in (RES_DIR, FIG_DIR, LABEL_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CLASSES = ["B", "IR", "OR", "N"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
CLASS3 = ["B", "IR", "OR"]

META_COLS = ["domain", "file_id", "group", "fs", "win_idx", "win_start", "rpm",
             "fault_type", "fault_size", "or_position", "load"]
DIM_TIME = ["t_mean_abs", "t_rms", "t_std", "t_var", "t_peak", "t_p2p"]

FEATURE_MAIN = "F2"
FEATURE_ALT = "F4"
FEATURE_ABLATION = ["F6", "F7", "F8"]

PROXY_TASKS = {
    "A_P2": dict(protocol="P2", classes=CLASSES,
                 desc="leave-one-load-out (source-internal)"),
    "A_P3": dict(protocol="P3", classes=CLASS3,
                 desc="cross sampling rate 12k<->48k"),
    "A_P4": dict(protocol="P4", classes=CLASS3,
                 desc="cross sensor position DE<->FE (closest to device change)"),
}
PROXY_ORDER = ["A_P2", "A_P3", "A_P4"]

METHODS_SHALLOW = ["M0", "M1", "M2", "M3", "M4"]
METHODS_PSEUDO = ["M5", "M6", "M7"]
METHODS_DEEP = ["M8", "M9"]
METHODS_ALL = METHODS_SHALLOW + METHODS_PSEUDO + ["M10"]

METHOD_DESC = {
    "M0": "no adaptation (source model applied directly)",
    "M1": "per-domain standardisation",
    "M2": "CORAL (covariance alignment)",
    "M3": "subspace alignment (SA)",
    "M4": "linear MMD minimisation",
    "M5": "pseudo-label self-training (entropy + confidence filter)",
    "M6": "CORAL + pseudo-label self-training",
    "M7": "class-balanced optimal-transport pseudo labels (Sinkhorn)",
    "M8": "DANN (adversarial domain adaptation)",
    "M9": "LMMD (class-conditional subdomain alignment)",
    "M10": "consensus soft-voting over M0-M7",
}

BASE_MODEL = "hgb"
SEEDS = [0, 1, 2]
RPM_TARGET = 577.2
RPM_TOL_REL = 0.05

PSEUDO_CONF_THRESHOLD = 0.70
PSEUDO_MAX_ITER = 5
PSEUDO_TOPK_PER_CLASS = None
USE_CLASS_PRIOR = True
SINKHORN_EPS = 0.05
SINKHORN_ITERS = 60

CONSENSUS_TOP_K = 3
MIN_ACCEPT_GAIN = 0.10
