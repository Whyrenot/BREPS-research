#!/usr/bin/env bash
# setup_repo.sh
# =============
# Full setup of BREPS on a fresh GPU box: the base environment (SAM1 /
# SAM2.1 / SAM-HQ / MobileSAM / ... -- torch 1.13.1, pydiffvg from source,
# per INSTALL_FIXES.md) plus two extra conda envs for backends that CANNOT
# share the base env:
#
#   base    torch==1.13.1 (pinned -- newer torch trips "CUDA driver is too
#           old" on these boxes, see INSTALL_FIXES.md). Everything this repo
#           already supports lives here.
#   sam_hq2 SysCV/sam-hq (sam-hq2 subproject). Its package is ALSO literally
#           named `sam2`, colliding with the official facebookresearch/sam2
#           package installed in `base` for SAM2.1 -- needs torch>=2.3.1.
#   sam3    facebookresearch/sam3. Needs Python>=3.12, torch>=2.7 -- hard
#           incompatible with `base`'s torch 1.13.1 pin. NOTE: SAM3 is only
#           INSTALLED here, not wired into the gradient-refine pipeline yet
#           (heatmaps/comp_hw_smoothed.load_sam3_model raises
#           NotImplementedError until someone confirms, on this env, that
#           SAM3's model exposes a prompt_encoder/mask_decoder pair usable
#           for autograd through the box prompt).
#
# scripts/*.py that accept --model_name auto-detect SAM-HQ2/SAM3 and
# re-exec themselves into the right env via `conda run` (see
# heatmaps/env_dispatch.py) -- you do not need to activate envs by hand,
# just make sure this script has created them.
#
# Usage:
#   ./scripts/setup_repo.sh                    # all three envs
#   ./scripts/setup_repo.sh --envs base         # only the base env
#   ./scripts/setup_repo.sh --envs base,sam_hq2 # base + SAM-HQ2
#   ./scripts/setup_repo.sh --vendor-dir ../vendor --cuda-tag cu121
#
# Idempotent: re-running skips a conda env that already exists (use
# --recreate to force-recreate) and steps that already succeeded (pydiffvg
# clone, pip installs) are cheap no-ops on a second run.

set -euo pipefail

# ---------------------------------------------------------------------------
# defaults / CLI
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$(cd "$REPO_ROOT/.." && pwd)"   # sibling dir, matches ../MODEL_CHECKPOINTS convention

ENVS="base,sam_hq2,sam3"
BASE_ENV_NAME="breps"
HQ2_ENV_NAME="sam_hq2"
SAM3_ENV_NAME="sam3"

BASE_PY="3.10"
HQ2_PY="3.10"
SAM3_PY="3.12"

BASE_CUDA_TAG="cu117"          # torch 1.13.1 wheel index
HQ2_CUDA_TAG="cu121"           # torch>=2.3.1 wheel index
SAM3_CUDA_TAG="cu128"          # matches facebookresearch/sam3 README

RECREATE=0

usage() {
    grep '^#' "${BASH_SOURCE[0]}" | sed -e 's/^#//' -e 's/^ //'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --envs) ENVS="$2"; shift 2 ;;
        --vendor-dir) VENDOR_DIR="$2"; shift 2 ;;
        --base-env-name) BASE_ENV_NAME="$2"; shift 2 ;;
        --hq2-env-name) HQ2_ENV_NAME="$2"; shift 2 ;;
        --sam3-env-name) SAM3_ENV_NAME="$2"; shift 2 ;;
        --cuda-tag) BASE_CUDA_TAG="$2"; HQ2_CUDA_TAG="$2"; SAM3_CUDA_TAG="$2"; shift 2 ;;
        --recreate) RECREATE=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown arg: $1" >&2; usage ;;
    esac
done

log()  { echo -e "\n\033[1;34m[setup_repo]\033[0m $*"; }
warn() { echo -e "\n\033[1;33m[setup_repo][warn]\033[0m $*" >&2; }

conda_env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

maybe_recreate() {
    local name="$1"
    if conda_env_exists "$name"; then
        if [[ "$RECREATE" == "1" ]]; then
            log "Removing existing env '$name' (--recreate)"
            conda env remove -n "$name" -y
            return 1  # "does not exist" -> caller (re)creates it
        fi
        return 0  # exists, keep it
    fi
    return 1  # does not exist
}

# ---------------------------------------------------------------------------
# base env: SAM1 / SAM2.1 / SAM-HQ / MobileSAM / ... (torch 1.13.1)
# mirrors INSTALL_FIXES.md step by step.
# ---------------------------------------------------------------------------
setup_base_env() {
    log "=== base env '$BASE_ENV_NAME' (torch 1.13.1) ==="

    if maybe_recreate "$BASE_ENV_NAME"; then
        log "conda env '$BASE_ENV_NAME' already exists, reusing (use --recreate to rebuild)"
    else
        conda create -n "$BASE_ENV_NAME" "python=$BASE_PY" -y
    fi

    # 1.1 / 1.2 -- pytorch + system deps (INSTALL_FIXES.md #1)
    conda run -n "$BASE_ENV_NAME" pip install torch==1.13.1 torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/${BASE_CUDA_TAG}"
    conda install -n "$BASE_ENV_NAME" -y scikit-image
    conda install -n "$BASE_ENV_NAME" -y -c anaconda cmake

    # 2 -- pydiffvg from source (INSTALL_FIXES.md #2)
    local diffvg_dir="$VENDOR_DIR/diffvg"
    if [[ ! -d "$diffvg_dir" ]]; then
        log "Cloning diffvg -> $diffvg_dir"
        git clone https://github.com/BachiLi/diffvg.git "$diffvg_dir"
        (cd "$diffvg_dir" && git submodule update --init --recursive)
    else
        log "diffvg already present at $diffvg_dir, skipping clone"
    fi
    # diffvg vendors an old pybind11 (cmake_minimum_required < 3.5); modern
    # CMake (>=3.31) refuses to configure that outright. CMAKE_POLICY_VERSION_MINIMUM
    # is CMake's own documented escape hatch for exactly this (old vendored
    # subdirectory, no local edits to diffvg's checkout needed).
    (cd "$diffvg_dir" && CMAKE_POLICY_VERSION_MINIMUM=3.5 conda run -n "$BASE_ENV_NAME" python setup.py install)
    conda run -n "$BASE_ENV_NAME" pip install svgpathtools cssutils

    # 3 -- pyproject.toml packaging fix: already committed in this repo's
    # working tree (see [tool.setuptools.packages.find] below); just verify.
    if ! grep -q "tool.setuptools.packages.find" "$REPO_ROOT/pyproject.toml"; then
        warn "pyproject.toml is missing [tool.setuptools.packages.find] -- expected it" \
             "already applied per INSTALL_FIXES.md #3. Appending it now."
        cat >> "$REPO_ROOT/pyproject.toml" <<'EOF'

[tool.setuptools.packages.find]
where = ["."]
include = ["isegm*"]
EOF
    fi

    # 4 -- private-dataset imports: already commented out in this repo's
    # working tree (isegm/data/datasets/__init__.py, isegm/inference/utils.py,
    # isegm/inference/predictors/__init__.py) per INSTALL_FIXES.md #4. If you
    # are setting up from a clean upstream checkout instead of this working
    # tree, redo those edits by hand before continuing.

    # 5 -- finish install
    conda run -n "$BASE_ENV_NAME" pip install loguru numba
    (cd "$REPO_ROOT" && conda run -n "$BASE_ENV_NAME" pip install -e .)

    log "Verifying base env ..."
    (cd "$REPO_ROOT" && conda run -n "$BASE_ENV_NAME" python3 scripts/evaluate_boxes_model_sam.py --help >/dev/null) \
        && log "base env OK" || warn "base env smoke-test failed -- check the log above"
}

# ---------------------------------------------------------------------------
# sam_hq2 env: SysCV/sam-hq (sam-hq2 subproject), torch>=2.3.1.
# Its package is named `sam2`, so it must NOT share an env with the base
# SAM2.1 install. Also needs this repo's `heatmaps`/`scripts` runtime deps
# (but NOT the full pinned-torch isegm install) so that scripts re-exec'd
# here via env_dispatch.py can still import heatmaps.* and segment_anything.
# ---------------------------------------------------------------------------
setup_samhq2_env() {
    log "=== SAM-HQ2 env '$HQ2_ENV_NAME' (torch>=2.3.1) ==="

    if maybe_recreate "$HQ2_ENV_NAME"; then
        log "conda env '$HQ2_ENV_NAME' already exists, reusing (use --recreate to rebuild)"
    else
        conda create -n "$HQ2_ENV_NAME" "python=$HQ2_PY" -y
    fi

    conda run -n "$HQ2_ENV_NAME" pip install torch==2.3.1 torchvision \
        --index-url "https://download.pytorch.org/whl/${HQ2_CUDA_TAG}"

    local samhq_dir="$VENDOR_DIR/sam-hq"
    if [[ ! -d "$samhq_dir" ]]; then
        log "Cloning SysCV/sam-hq -> $samhq_dir"
        git clone https://github.com/SysCV/sam-hq.git "$samhq_dir"
    else
        log "sam-hq already present at $samhq_dir, skipping clone"
    fi
    (cd "$samhq_dir/sam-hq2" && conda run -n "$HQ2_ENV_NAME" pip install -e .)

    # Minimal runtime deps for heatmaps/*.py + scripts/*.py (NOT the full
    # torch==1.13.1-pinned isegm package -- deliberately no `pip install -e .`
    # here, that would fight the sam_hq2 torch version).
    conda run -n "$HQ2_ENV_NAME" pip install \
        numpy pandas "opencv-python-headless>=4.8.1.78" "matplotlib>=3.10.7" \
        scipy loguru tqdm pyyaml "segment-anything>=1.0"

    log "sam_hq2 env ready. Smoke-test once you have a checkpoint:"
    log "  conda run -n $HQ2_ENV_NAME python -c \"from sam2.build_sam import build_sam2; print('sam2 (HQ2) import OK')\""
}

# ---------------------------------------------------------------------------
# sam3 env: facebookresearch/sam3, Python>=3.12, torch>=2.7.
# NOT wired into the gradient-refine pipeline yet -- installed for later use
# once someone verifies SAM3's box-prompt / autograd API on this env (see
# heatmaps/comp_hw_smoothed.load_sam3_model).
# ---------------------------------------------------------------------------
setup_sam3_env() {
    log "=== SAM3 env '$SAM3_ENV_NAME' (Python $SAM3_PY, torch>=2.7) ==="
    log "NOTE: install-only. load_sam3_model() in heatmaps/comp_hw_smoothed.py"
    log "      still raises NotImplementedError -- see that docstring before"
    log "      trying to run refine_box_iou_grad.py --model_name SAM3."

    if maybe_recreate "$SAM3_ENV_NAME"; then
        log "conda env '$SAM3_ENV_NAME' already exists, reusing (use --recreate to rebuild)"
    else
        conda create -n "$SAM3_ENV_NAME" "python=$SAM3_PY" -y
    fi

    conda run -n "$SAM3_ENV_NAME" pip install torch==2.10.0 torchvision \
        --index-url "https://download.pytorch.org/whl/${SAM3_CUDA_TAG}"

    # facebookresearch/sam3 checkpoints are gated on Hugging Face -- request
    # access first (see https://github.com/facebookresearch/sam3) and run
    # `huggingface-cli login` in this env before downloading checkpoints.
    conda run -n "$SAM3_ENV_NAME" pip install sam3

    conda run -n "$SAM3_ENV_NAME" pip install \
        numpy pandas "opencv-python-headless>=4.8.1.78" "matplotlib>=3.10.7" \
        scipy loguru tqdm pyyaml "segment-anything>=1.0"

    log "sam3 env ready (install-only)."
    log "  conda run -n $SAM3_ENV_NAME python -c \"from sam3.model_builder import build_sam3_image_model; print('sam3 import OK')\""
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
IFS=',' read -ra WANTED <<< "$ENVS"
for env_key in "${WANTED[@]}"; do
    case "$env_key" in
        base)    setup_base_env ;;
        sam_hq2) setup_samhq2_env ;;
        sam3)    setup_sam3_env ;;
        *) echo "Unknown --envs entry: $env_key (expected base,sam_hq2,sam3)" >&2; exit 1 ;;
    esac
done

log "Done. Envs created: $ENVS"
log "Run e.g.:"
log "  conda run -n $BASE_ENV_NAME python scripts/refine_box_iou_grad.py --model_name SAM ..."
log "  conda run -n $BASE_ENV_NAME python scripts/refine_box_iou_grad.py --model_name SAM-HQ2 ...   # auto re-execs into '$HQ2_ENV_NAME'"
