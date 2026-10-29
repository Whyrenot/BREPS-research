#!/usr/bin/env bash
# The steering site scan on the user study for SAM-1, SAM-2.1 and SAM-3, one
# after the other on one GPU, each followed by its HTML report:
#   scripts/steer_models_scan.py    -> exp_res/steer_scan_<model>/per_case_mm{0,1}.csv.gz, ...
#   scripts/steer_models_report.py  -> exp_res/steer_scan_<model>/steer_scan_report.html, summary.md
# SAM-3 re-execs itself into the sam3 conda env (heatmaps/env_dispatch.py).
# A failing model does not stop the next one; the exit code says whether any failed.
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_steer_models_scan.sh
#
#   MODELS="SAM2.1" ...      only some models
#   ALPHAS=0.5,1,...,32 ...  another alpha grid (default: the scan's 0.5,1,2,4,8,12,16)
#   SUFFIX=_a32 ...          appended to the out dirs (e.g. for another grid)
#   SMOKE=1 ...              2 images per half / subset, alphas 1 and 4, out dirs *_smoke
set -uo pipefail
cd "$(dirname "$0")/.."

ROOT=${ROOT:-/home/jovyan/shares/SR006.nfs2/pishugin/user_study/FOR_TEST}
CKPTS=${CKPTS:-/home/jovyan/shares/SR006.nfs2/pishugin/temp/rclicks/MODEL_CHECKPOINTS}
CKPT_SAM=${CKPT_SAM:-$CKPTS/SAM/sam_vit_b_01ec64.pth}
CKPT_SAM21=${CKPT_SAM21:-$CKPTS/SAM2.1/sam2.1_hiera_base_plus.pt}
# results/sam21full_m.csv was run on hiera_large, so it is no reference for
# base_plus: no repro check / comparison rows unless a base_plus csv is given
REF_SAM21=${REF_SAM21:-}
CKPT_SAM3=${CKPT_SAM3:-}            # "" = the builder downloads / reuses facebook/sam3 from the HF cache
MODELS=${MODELS:-"SAM SAM2.1 SAM3"}
SMOKE=${SMOKE:-0}
status=0

ALPHAS=${ALPHAS:-}
extra=()
suffix=${SUFFIX:-}
[ -n "$ALPHAS" ] && extra=(--alphas "$ALPHAS")
if [ "$SMOKE" = "1" ]; then
    extra=(--limit_images 2 --alphas 1,4)
    suffix="${suffix}_smoke"
fi

for model in $MODELS; do
    case "$model" in
        SAM)    ckpt=$CKPT_SAM;   ref=results/samfull_m.csv ;;
        SAM2.1) ckpt=$CKPT_SAM21; ref=$REF_SAM21 ;;
        SAM3)   ckpt=$CKPT_SAM3;  ref=results/sam3full_m.csv ;;
        *)      echo "unknown model $model"; status=1; continue ;;
    esac
    # sam3full_m.csv was made one box per call with TF32 matmuls on (sam3 turns
    # them on at import); this scan decodes a whole image per call in fp32, so
    # boxes near the threshold differ: the mean must agree, a few boxes may not
    rep=()
    [ "$model" = "SAM3" ] && rep=(--repro_max_frac 0.2)
    out=exp_res/steer_scan_${model}${suffix}
    mkdir -p "$out"
    echo "=== $model scan $(date)"
    python scripts/steer_models_scan.py --model_name "$model" --root "$ROOT" --checkpoint_path "$ckpt" \
        --reference_csv "$ref" --out_dir "$out" "${rep[@]}" "${extra[@]}" 2>&1 | tee "$out/run.log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "$model scan FAILED"; status=1; continue; fi
    echo "=== $model report $(date)"
    python scripts/steer_models_report.py --run_dir "$out" --reference_csv "$ref" 2>&1 | tee "$out/report.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "$model report FAILED"; status=1; }
done

echo "=== done $(date), status $status"
exit $status
