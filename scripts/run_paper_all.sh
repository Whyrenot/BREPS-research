#!/usr/bin/env bash
# Everything of the paper's user-study experiment for one annotation set, in one go:
#   1. the candidate pool of SAM, SAM2.1, SAM3 (build_candidate_pool.py, images split over GPUS)
#        -> exp_res/pool_<model>${TAG}/
#   2. CatBoost on every pool, out of fold         -> exp_res/cb_<model>${TAG}/
#   3. the TRANSFER_FROM model's selectors on the others -> exp_res/cb_<src>_on_<model>${TAG}/
#   4. one CSV with every model x annotation, every method and feature
#        -> exp_res/paper_table${TAG}.csv + paper_table${TAG}_columns.md
# Default USE=mp: the mobile (m) and the desktop (p) annotations; TAG=_mp, so the
# mobile-only results (pool_<model>, cb_<model>) are left alone and joined into the
# table as a reference. The steering vectors are the scans' (exp_res/steer_scan_<model>,
# mobile annotations of COCO A). A failing step does not stop the next; the table is
# built from whatever finished. The log goes to exp_res/paper_all${TAG}.log too.
#
#   GPUS="0 1 2 3 4 5 6 7" bash scripts/run_paper_all.sh
#   SMOKE=1 ...            2 images per subset, CatBoost 2 folds / 50 iterations, dirs *_smoke
#   MODELS="SAM3" ...      only some models (the transfer needs TRANSFER_FROM among them or done before)
#   STEPS="cb table" ...   only some steps of: pool cb table
set -uo pipefail
cd "$(dirname "$0")/.."

export USE=${USE:-mp}
export TAG=${TAG-$([ "$USE" = "m" ] && echo "" || echo "_$USE")}
export MODELS=${MODELS:-"SAM SAM2.1 SAM3"}
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}
export GPUS=${GPUS//,/ }
TRANSFER_FROM=${TRANSFER_FROM-SAM3}
STEPS=${STEPS:-"pool cb table"}
SMOKE=${SMOKE:-0}
export SMOKE
read -r -a gpus <<< "$GPUS"
suffix=""; [ "$SMOKE" = "1" ] && suffix="_smoke"
LOG=exp_res/paper_all${TAG}${suffix}.log
mkdir -p exp_res
exec > >(tee -a "$LOG") 2>&1
status=0
echo "##### run_paper_all: USE=$USE TAG=$TAG MODELS=$MODELS GPUS=${gpus[*]} TRANSFER_FROM=$TRANSFER_FROM SMOKE=$SMOKE $(date)"

if [[ " $STEPS " == *" pool "* ]]; then
    echo "##### 1/3 candidate pools $(date)"
    bash scripts/run_candidate_pool.sh || status=1
fi

if [[ " $STEPS " == *" cb "* ]]; then
    echo "##### 2/3 CatBoost and transfer $(date)"
    GPU=${gpus[0]} TRANSFER_FROM=$TRANSFER_FROM bash scripts/run_pool_catboost.sh || status=1
fi

if [[ " $STEPS " == *" table "* ]]; then
    echo "##### 3/3 the analysis table $(date)"
    python scripts/build_paper_table.py --tag "${TAG}${suffix}" --models $MODELS --transfer_from "$TRANSFER_FROM" \
        --mobile_tag "$([ "$USE" = "m" ] && echo none || echo "$suffix")" --out "exp_res/paper_table${TAG}${suffix}" \
        || status=1
fi

echo "##### done $(date), status $status (log: $LOG)"
exit $status
