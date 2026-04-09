"""
config.py  —  Single source of truth for all paths and settings.

If you are using Docker (recommended), BASE_DIR = '/app' and you
never need to change anything.

If you are running locally without Docker, change BASE_DIR to your
repo root, e.g.:
    Windows : BASE_DIR = r"D:/Lightweight-AI-based-Vision-System-for-Smart-Parking-Application"
    Mac/Linux: BASE_DIR = "/home/yourname/repo"
"""

import os

# ── Base directory ────────────────────────────────────────────────────────────
# Docker users: leave as '/app'  (this is always correct inside the container)
# Local users : change to your repo root path
BASE_DIR = os.environ.get("PROJECT_ROOT", "/app")

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME      = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
INPUT_SIZE      = 320
NUM_CLASSES     = 2
CLASSES         = ["space-empty", "space-occupied"]

# ── TF OD API ─────────────────────────────────────────────────────────────────
TF_MODELS_DIR   = os.path.join(BASE_DIR, "tensorflow_models")
RESEARCH_DIR    = os.path.join(TF_MODELS_DIR, "research")
OD_DIR          = os.path.join(RESEARCH_DIR, "object_detection")

# ── Pretrained checkpoint ─────────────────────────────────────────────────────
PRETRAINED_DIR  = os.path.join(BASE_DIR, "models", "pretrained", MODEL_NAME)
PRETRAINED_CKPT = os.path.join(PRETRAINED_DIR, "checkpoint", "ckpt-0")

# ── Datasets ──────────────────────────────────────────────────────────────────
DATASETS = {
    "pklot": {
        "raw_dir"       : os.path.join(BASE_DIR, "datasets", "pklot", "raw"),
        "annotations_dir": os.path.join(BASE_DIR, "datasets", "pklot", "annotations"),
        "tfrecord_dir"  : os.path.join(BASE_DIR, "datasets", "pklot", "tfrecords"),
        "label_map"     : os.path.join(BASE_DIR, "datasets", "pklot", "label_map.pbtxt"),
        "pipeline_cfg"  : os.path.join(BASE_DIR, "configs", "pklot", "pipeline.config"),
        "model_dir"     : os.path.join(BASE_DIR, "models", "pklot"),
    },
    "custom": {
        "raw_dir"       : os.path.join(BASE_DIR, "datasets", "custom", "raw"),
        "annotations_dir": os.path.join(BASE_DIR, "datasets", "custom", "annotations"),
        "tfrecord_dir"  : os.path.join(BASE_DIR, "datasets", "custom", "tfrecords"),
        "label_map"     : os.path.join(BASE_DIR, "datasets", "custom", "label_map.pbtxt"),
        "pipeline_cfg"  : os.path.join(BASE_DIR, "configs", "custom", "pipeline.config"),
        "model_dir"     : os.path.join(BASE_DIR, "models", "custom"),
    },
}

# ── Results ───────────────────────────────────────────────────────────────────
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
METRICS_DIR     = os.path.join(RESULTS_DIR, "metrics")
VIZ_DIR         = os.path.join(RESULTS_DIR, "visualizations")

# ── Quick path check (run this file directly to verify) ──────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append(RESEARCH_DIR)

    checks = {
        "BASE_DIR"      : BASE_DIR,
        "TF_MODELS_DIR" : TF_MODELS_DIR,
        "OD_DIR"        : OD_DIR,
        "PRETRAINED_CKPT": PRETRAINED_CKPT + ".index",
    }
    for ds in DATASETS.values():
        checks[f"label_map ({ds['label_map']})"] = ds["label_map"]
        checks[f"pipeline ({ds['pipeline_cfg']})"] = ds["pipeline_cfg"]
        checks[f"tfrecords ({ds['tfrecord_dir']})"] = ds["tfrecord_dir"]

    print(f"\n{'='*55}")
    print(f"  PROJECT_ROOT = {BASE_DIR}")
    print(f"{'='*55}")
    all_ok = True
    for name, path in checks.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌  MISSING"
        print(f"  {status}  {name}")
        if not exists:
            all_ok = False
    print(f"{'='*55}")
    print(f"  {'All paths OK!' if all_ok else 'Fix missing paths above.'}")
