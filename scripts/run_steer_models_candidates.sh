#!/usr/bin/env bash
# Regressor features and CatBoost candidates of the L2.norm2 steering vector, for
# SAM-1, SAM-2.1 and SAM-3, one after the other on one GPU; each needs its
# finished scan (run_steer_models_scan.sh -> exp_res/steer_scan_<model>):
#   scripts/steer_models_candidates.py -> exp_res/steer_cand_<model>/rows.csv.gz, candidates_mm{0,1}.csv
#   scripts/steer_multimask.py --from_dump -> exp_res/steer_cand_<model>/gbm/summary.md (alpha regressors, CPU)
# SAM-3 re-execs itself into the sam3 conda env. A failing model does not stop the next one.
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_steer_models_candidates.sh
#   MODELS="SAM3" ...   only some models;  SMOKE=1 ...  2 images per half / subset, out dirs *_smoke
#   SITES="L2.norm3 hyper" ...  steering sites (default L2.norm3), one out dir per site
set -uo pipefail
cd "$(dirname "$0")/.."

ROOT=${ROOT:-/home/jovyan/shares/SR006.nfs2/pishugin/user_study/FOR_TEST}
CKPTS=${CKPTS:-/home/jovyan/shares/SR006.nfs2/pishugin/temp/rclicks/MODEL_CHECKPOINTS}
CKPT_SAM=${CKPT_SAM:-$CKPTS/SAM/sam_vit_b_01ec64.pth}
CKPT_SAM21=${CKPT_SAM21:-$CKPTS/SAM2.1/sam2.1_hiera_base_plus.pt}
CKPT_SAM3=${CKPT_SAM3:-}
REF_SAM21=${REF_SAM21:-}          # results/sam21full_m.csv is hiera_large: no reference for base_plus
MODELS=${MODELS:-"SAM SAM2.1 SAM3"}
ALPHAS=${ALPHAS:-1,2,3,4,6,8,12,16,24,32}
SITES=${SITES:-L2.norm3}
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
    if [ ! -f "$scan/vectors.npz" ]; then echo "$model: no scan in $scan, run run_steer_models_scan.sh first"; status=1; continue; fi
    for site in $SITES; do
        out=exp_res/steer_cand_${model}_${site}${suffix}
        mkdir -p "$out"
        echo "=== $model $site candidates $(date)"
        python scripts/steer_models_candidates.py --model_name "$model" --root "$ROOT" --checkpoint_path "$ckpt" \
            --scan_dir "$scan" --site "$site" --reference_csv "$ref" --alphas "$ALPHAS" --out_dir "$out" \
            "${extra[@]}" 2>&1 | tee "$out/run.log"
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "$model $site candidates FAILED"; status=1; continue; fi
        echo "=== $model $site alpha regressors $(date)"
        python scripts/steer_multimask.py --from_dump "$out/rows.csv.gz" --alphas "$ALPHAS" --fixed_alphas 1,4,12 \
            --reference_csv "$ref" --out_dir "$out/gbm" 2>&1 | tee "$out/gbm.log"
        [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "$model $site regressors FAILED"; status=1; }
    done
done

echo "=== done $(date), status $status"
exit $status
