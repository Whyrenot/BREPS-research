#!/usr/bin/env bash
# The CatBoost selector over the paper's candidate pools (scripts/train_pool_catboost.py):
#   exp_res/pool_<model>/  ->  exp_res/cb_<model>/   out of fold over 5 image folds, COCO A
#                                                    excluded; fold models + a model on all images
# and, with TRANSFER_FROM, that model's selectors applied to the other models' pools:
#   exp_res/cb_<src>_on_<model>/   (fold models: neither the image nor this model's candidates were seen)
# No torch: any env with pandas, scikit-learn and catboost (the Linux wheels carry GPU support).
# A failing model does not stop the next one.
#
#   GPU=0 bash scripts/run_pool_catboost.sh
#   MODELS="SAM3" ...          only some models
#   TRANSFER_FROM=SAM3 ...     also apply the SAM3 selectors to the other MODELS (after training them all)
#   TASK_TYPE=CPU THREADS=32 ...    on the CPU
#   SMOKE=1 ...                exp_res/pool_<model>_smoke, 2 folds, 50 iterations, out dirs *_smoke
#   TAG=_mp ...                the pools of another annotation set: pool_<model>_mp -> cb_<model>_mp,
#                              cb_<src>_on_<model>_mp
set -uo pipefail
cd "$(dirname "$0")/.."

MODELS=${MODELS:-"SAM SAM2.1 SAM3"}
TRANSFER_FROM=${TRANSFER_FROM:-}
GPU=${GPU:-0}
TASK_TYPE=${TASK_TYPE:-GPU}
THREADS=${THREADS:--1}
ITER=${ITER:-800}
SMOKE=${SMOKE:-0}
TAG=${TAG:-}
status=0

extra=(--task_type "$TASK_TYPE" --threads "$THREADS")
suffix=""
if [ "$SMOKE" = "1" ]; then
    extra+=(--folds 2 --n_boot 200)
    ITER=50
    suffix="_smoke"
fi
extra+=(--iterations "$ITER")
# CatBoost addresses the GPUs it can see: show it one and call it device 0
export CUDA_VISIBLE_DEVICES=$GPU

for model in $MODELS; do
    pool=exp_res/pool_${model}${TAG}${suffix}
    out=exp_res/cb_${model}${TAG}${suffix}
    if [ ! -f "$pool/pool.csv.gz" ]; then echo "$model: no merged pool in $pool"; status=1; continue; fi
    mkdir -p "$out"
    echo "=== $model CatBoost $pool -> $out $(date)"
    python scripts/train_pool_catboost.py --pool_dir "$pool" --out_dir "$out" --devices 0 "${extra[@]}" \
        2>&1 | tee "$out/run.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "$model CatBoost FAILED"; status=1; }
done

if [ -n "$TRANSFER_FROM" ]; then
    src=exp_res/cb_${TRANSFER_FROM}${TAG}${suffix}
    for model in $MODELS; do
        [ "$model" = "$TRANSFER_FROM" ] && continue
        pool=exp_res/pool_${model}${TAG}${suffix}
        out=exp_res/cb_${TRANSFER_FROM}_on_${model}${TAG}${suffix}
        if [ ! -f "$src/model_full.cbm" ] || [ ! -f "$pool/pool.csv.gz" ]; then
            echo "transfer $TRANSFER_FROM -> $model: need $src/model_full.cbm and $pool/pool.csv.gz"; status=1; continue
        fi
        mkdir -p "$out"
        echo "=== transfer $TRANSFER_FROM -> $model $pool -> $out $(date)"
        python scripts/train_pool_catboost.py --apply_model "$src" --pool_dir "$pool" --out_dir "$out" --devices 0 \
            "${extra[@]}" 2>&1 | tee "$out/run.log"
        [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "transfer $TRANSFER_FROM -> $model FAILED"; status=1; }
    done
fi

echo "=== done $(date), status $status"
exit $status
