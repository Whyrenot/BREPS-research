#!/usr/bin/env bash
# The paper's candidate pool for SAM-1, SAM-2.1 and SAM-3, one after the other on
# one GPU (scripts/build_candidate_pool.py -> exp_res/pool_<model>/pool.csv.gz,
# per_box.csv.gz, rows.csv.gz): undef, head-select, grad 30 steps with and
# without multimask, BoN with centre noise with and without multimask, steering
# at L2.norm3 at every alpha with and without multimask. Each model needs its
# finished scan (exp_res/steer_scan_<model>: the steering vector). SAM-3
# re-execs itself into the sam3 conda env. A failing model does not stop the next.
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_candidate_pool.sh
#   MODELS="SAM3" ...   only some models;  SMOKE=1 ...  2 images per half / subset, out dirs *_smoke
set -uo pipefail
cd "$(dirname "$0")/.."

ROOT=${ROOT:-/home/jovyan/shares/SR006.nfs2/pishugin/user_study/FOR_TEST}
CKPTS=${CKPTS:-/home/jovyan/shares/SR006.nfs2/pishugin/temp/rclicks/MODEL_CHECKPOINTS}
CKPT_SAM=${CKPT_SAM:-$CKPTS/SAM/sam_vit_b_01ec64.pth}
CKPT_SAM21=${CKPT_SAM21:-$CKPTS/SAM2.1/sam2.1_hiera_base_plus.pt}
CKPT_SAM3=${CKPT_SAM3:-}
REF_SAM21=${REF_SAM21:-}          # results/sam21full_m.csv is hiera_large: no reference for base_plus
MODELS=${MODELS:-"SAM SAM2.1 SAM3"}
SITE=${SITE:-L2.norm3}
SMOKE=${SMOKE:-0}
status=0

extra=()
suffix=""
if [ "$SMOKE" = "1" ]; then
    extra=(--limit_images 2)
    suffix="_smoke"
fi

for model in $MODELS; do
    case "$model" in
        SAM)    ckpt=$CKPT_SAM;   ref=results/samfull_m.csv ;;
        SAM2.1) ckpt=$CKPT_SAM21; ref=$REF_SAM21 ;;
        SAM3)   ckpt=$CKPT_SAM3;  ref=results/sam3full_m.csv ;;
        *)      echo "unknown model $model"; status=1; continue ;;
    esac
    scan=exp_res/steer_scan_${model}
    if [ ! -f "$scan/vectors.npz" ]; then echo "$model: no scan in $scan"; status=1; continue; fi
    out=exp_res/pool_${model}${suffix}
    mkdir -p "$out"
    echo "=== $model pool $(date)"
    python scripts/build_candidate_pool.py --model_name "$model" --root "$ROOT" --checkpoint_path "$ckpt" \
        --scan_dir "$scan" --site "$SITE" --reference_csv "$ref" --out_dir "$out" "${extra[@]}" 2>&1 | tee "$out/run.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "$model pool FAILED"; status=1; }
done

echo "=== done $(date), status $status"
exit $status
