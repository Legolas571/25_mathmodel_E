import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent

TASK1_DIR = ROOT / "data_anls_and_feature_extraction"
T1_OUT = TASK1_DIR / "outputs"

SRC_FEATURES = T1_OUT / "features_source.csv"
TGT_FEATURES = T1_OUT / "features_target.csv"
T1_SPLITS = T1_OUT / "splits.json"
AUDIT_SOURCE = T1_OUT / "audit_source.csv"
FAULT_ORDERS = T1_OUT / "fault_orders.csv"
WINDOWS_SRC = T1_OUT / "windows_source_signal.npz"
WINDOWS_TGT = T1_OUT / "windows_target_signal.npz"

RES_DIR = BASE_DIR / "results"
FIG_DIR = BASE_DIR / "figures"
MODEL_DIR = BASE_DIR / "models"
for _d in (RES_DIR, FIG_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CLASSES = ["B", "IR", "OR", "N"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
CLASS3 = ["B", "IR", "OR"]

META_COLS = ["domain", "file_id", "group", "fs", "win_idx", "win_start", "rpm",
             "fault_type", "fault_size", "or_position", "load"]
DIM_TIME = ["t_mean_abs", "t_rms", "t_std", "t_var", "t_peak", "t_p2p"]

PROTOCOLS = ["P1", "P2", "P3", "P4", "P5"]
FOLDS = {"P1": 5, "P2": 4, "P3": 2, "P4": 2, "P5": 5}
PROTOCOL_CLASSES = {"P1": CLASSES, "P2": CLASSES, "P3": CLASS3, "P4": CLASS3, "P5": CLASSES}
PROTOCOL_DESC = {
    "P1": "random file-level 5-fold (stratified by class)",
    "P2": "leave-one-load-out (4 folds by load)",
    "P3": "cross sampling rate 12k<->48k (3 classes)",
    "P4": "cross sensor position DE<->FE (3 classes)",
    "P5": "48 kHz only, 4 classes, 5-fold (sampling-rate clean control)",
}

MAIN_FEATURE_SETS = ["F1", "F2", "F3", "F6", "F7"]
ALL_FEATURE_SETS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
MAIN_MODELS = ["lr", "svm", "rf", "hgb"]
DEEP_MODELS = ["mlp", "vote"]

SEEDS = [0, 1, 2]
SCALER = "robust"
WINSORIZE = None
FOCAL_GAMMA = 2.0
N_JOBS = 1

CNN_WIN_POINTS = 1536
CNN_RESAMPLE_FS = 12000
