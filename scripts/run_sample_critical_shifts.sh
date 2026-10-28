#!/usr/bin/env bash
# Critical shifts on a 20-images-per-subset sample of the user study (all 10
# FOR_TEST subsets) for SAM-1, SAM-2.1 and SAM-3, one after the other, each on
# every visible GPU (scripts/sample_critical_shifts.py):
#   exp_res/critical_shifts_sample/<model>/critical_shifts_all.json, summary.csv, selection.json, ...
# First every model runs only its check (comp_hw's batched path vs the model's
# own predict(), plus a timing) -- minutes, so a broken model stops everything
# before the night instead of showing up in the morning. Then the full runs; a
# failing model does not stop the next one, the exit code says whether any
# failed. Re-running the same command resumes. SAM-3 re-execs itself into the
# sam3 conda env (heatmaps/env_dispatch.py). Run from the pishugin-breps env.
#
#   CUDA_VISIBLE_DEVICES=5,7 bash scripts/run_sample_critical_shifts.sh
#
#   SMOKE=1 ...                   one image per subset, no replacement, out dir *_smoke
#   MODELS="SAM3" ...             only some models
#   CUDA_VISIBLE_DEVICES=0,1 ...  the GPUs torchrun uses (default: all)
#   N_IMAGES=5 OUT=... ...        fewer images / another output dir
set -uo pipefail
cd "$(dirname "$0")/.."

ROOT=${ROOT:-/home/jovyan/shares/SR006.nfs2/pishugin/user_study/FOR_TEST}
CKPTS=${CKPTS:-/home/jovyan/shares/SR006.nfs2/pishugin/temp/rclicks/MODEL_CHECKPOINTS}
CKPT_SAM=${CKPT_SAM:-$CKPTS/SAM/sam_vit_b_01ec64.pth}
CKPT_SAM21=${CKPT_SAM21:-$CKPTS/SAM2.1/sam2.1_hiera_base_plus.pt}  # the size is read from the file name
CKPT_SAM3=${CKPT_SAM3:-}            # "" = the builder takes the weights from the HF cache
MODELS=${MODELS:-"SAM SAM2.1 SAM3"}
SMOKE=${SMOKE:-0}
BATCH=${BATCH:-512}
status=0

extra=()
if [ "$SMOKE" = "1" ]; then
    N_IMAGES=${N_IMAGES:-1}
    OUT=${OUT:-exp_res/critical_shifts_sample_smoke}
    extra=(--max_tried 1)
else
    N_IMAGES=${N_IMAGES:-20}
    OUT=${OUT:-exp_res/critical_shifts_sample}
fi

run() {  # model, extra args...
    local model=$1 ckpt
    shift
    case "$model" in
        SAM)    ckpt=$CKPT_SAM ;;
        SAM2.1) ckpt=$CKPT_SAM21 ;;
        SAM3)   ckpt=$CKPT_SAM3 ;;
    esac
    python scripts/sample_critical_shifts.py --model_name "$model" --root "$ROOT" \
        --checkpoint_path "$ckpt" --out_dir "$OUT/$model" \
        --n_images "$N_IMAGES" --batch_size "$BATCH" "${extra[@]}" "$@"
}

for model in $MODELS; do
    case "$model" in
        SAM3)   ;;
        SAM)    [ -f "$CKPT_SAM" ] || { echo "SAM: no checkpoint $CKPT_SAM"; exit 1; } ;;
        SAM2.1) [ -f "$CKPT_SAM21" ] || { echo "SAM2.1: no checkpoint $CKPT_SAM21"; exit 1; } ;;
        *)      echo "unknown model $model"; exit 1 ;;
    esac
done

for model in $MODELS; do
    mkdir -p "$OUT/$model"
    echo "=== $model check $(date)"
    run "$model" --check_only 2>&1 | tee "$OUT/$model/check.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "$model check FAILED, nothing was run"; exit 1; }
done

for model in $MODELS; do
    echo "=== $model run $(date)"
    run "$model" --no_check 2>&1 | tee -a "$OUT/$model/run.log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "$model FAILED"; status=1; fi
done

echo "=== done $(date), status $status"
exit $status
