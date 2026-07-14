"""Default paths anchored to the selected Isaac-GR00T checkout."""

from __future__ import annotations

from groot_rlt.groot_repo import ensure_groot_repo

GROOT_REPO_ROOT = ensure_groot_repo()
L10_RAW_DATASET_ROOT = (
    GROOT_REPO_ROOT
    / "demo_data"
    / "l10_hand"
    / "rokae"
    / "mission1"
    / "groot_full_orientation_multiview_resync_20260509T122113Z_20260509T125003Z"
)
L10_TRIMMED_DATASET_DIR = (
    GROOT_REPO_ROOT / "outputs" / "IsaacLab" / "rokae" / "mission1" / "trimmed"
)
L10_PREPARED_DATASET_DIR = L10_TRIMMED_DATASET_DIR
L10_MODALITY_CONFIG_PATH = (
    GROOT_REPO_ROOT / "examples" / "IsaacLab" / "rokae_xmate3_l10_multiview_modality_config.py"
)
L10_MODEL_DIR = GROOT_REPO_ROOT / "checkpoints" / "rokae_xmate3_l10_full_orientation_overfit"
L10_VLM_MODEL_PATH = GROOT_REPO_ROOT / "checkpoints" / "nvidia" / "Cosmos-Reason2-2B"
L10_BASE_MODEL_PATH = GROOT_REPO_ROOT / "checkpoints" / "GR00T-N1.7-3B"
L10_INSTRUCTION = "pick up the bottle and place it in the box"
