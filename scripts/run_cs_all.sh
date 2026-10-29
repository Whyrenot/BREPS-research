#!/usr/bin/env bash
# The paper's methods and selectors on the critical shifts (the attacks of
# sample_critical_shifts.py), for SAM, SAM2.1, SAM3, in one go:
#   1. the candidate pool of every model on its own attacks: every case gives the
#      attack box (kind bad) and its good neighbour (kind best)
#        <CS_DIR>/<model>/critical_shifts_all.json -> exp_res/pool_<model>_cs/
#   2. CatBoost trained on the attacks, out of fold      -> exp_res/cb_<model>_cs/
#      and the SAM3 one on the other models' attacks     -> exp_res/cb_SAM3_on_<model>_cs/
#   3. the user-study selectors (US_TAG run, m + p) on the attacks, no retraining:
#      the model's own                                   -> exp_res/cb_<model>_mp_on_cs/
#      SAM3's on the other models                        -> exp_res/cb_SAM3_mp_on_<model>_cs/
#      (fold models: an image is scored by the user-study model of the fold that held it out)
#   4. one CSV, a row per model x box, every method, feature and selector
#        -> exp_res/paper_table_cs.csv + paper_table_cs_columns.md
# Steering: the scans' vectors and the user-study run's frozen alpha*. COCO A images
# (the vector's) are scored by no selector. A failing step does not stop the next.
# The log goes to exp_res/paper_cs.log too.
#
#   GPUS="0 1 2 3 4 5 6 7" bash scripts/run_cs_all.sh
#   CS_DIR=exp_res/critical_shifts_sample ...   another attack set (default exp_res/critical_shifts_full)
#   SMOKE=1 ...            2 images per subset, CatBoost 2 folds / 50 iterations, dirs *_smoke
#   STEPS="cb table" ...   only some steps of: pool cb us table
set -uo pipefail
cd "$(dirname "$0")/.."

export CS_DIR=${CS_DIR:-exp_res/critical_shifts_full}
export TAG=_cs
US_TAG=${US_TAG:-_mp}
export MODELS=${MODELS:-"SAM SAM2.1 SAM3"}
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}
export GPUS=${GPUS//,/ }
TRANSFER_FROM=${TRANSFER_FROM-SAM3}
STEPS=${STEPS:-"pool cb us table"}
SMOKE=${SMOKE:-0}
export SMOKE
TASK_TYPE=${TASK_TYPE:-GPU}
read -r -a gpus <<< "$GPUS"
sfx=""; [ "$SMOKE" = "1" ] && sfx="_smoke"
LOG=exp_res/paper_cs${sfx}.log
mkdir -p exp_res
exec > >(tee -a "$LOG") 2>&1
status=0
echo "##### run_cs_all: CS_DIR=$CS_DIR MODELS=$MODELS GPUS=${gpus[*]} US_TAG=$US_TAG TRANSFER_FROM=$TRANSFER_FROM SMOKE=$SMOKE $(date)"

if [[ " $STEPS " == *" pool "* ]]; then
    echo "##### 1/4 candidate pools on the attacks $(date)"
    bash scripts/run_candidate_pool.sh || status=1
fi

if [[ " $STEPS " == *" cb "* ]]; then
    echo "##### 2/4 CatBoost trained on the attacks, and its transfer $(date)"
    GPU=${gpus[0]} TRANSFER_FROM=$TRANSFER_FROM ALPHA_STAR_FROM="exp_res/cb_{model}${US_TAG}" \
        bash scripts/run_pool_catboost.sh || status=1
fi

if [[ " $STEPS " == *" us "* ]]; then
    echo "##### 3/4 the user-study selectors on the attacks $(date)"
    extra=(--task_type "$TASK_TYPE" --devices 0)
    [ "$SMOKE" = "1" ] && extra+=(--n_boot 200)
    for model in $MODELS; do
        pool=exp_res/pool_${model}_cs${sfx}
        if [ ! -f "$pool/pool.csv.gz" ]; then echo "$model: no pool in $pool"; status=1; continue; fi
        srcs=("$model")
        if [ -n "$TRANSFER_FROM" ] && [ "$TRANSFER_FROM" != "$model" ]; then srcs+=("$TRANSFER_FROM"); fi
        for src in "${srcs[@]}"; do
            if [ "$src" = "$model" ]; then out=exp_res/cb_${model}${US_TAG}_on_cs${sfx}
            else out=exp_res/cb_${src}${US_TAG}_on_${model}_cs${sfx}; fi
            if [ ! -f "exp_res/cb_${src}${US_TAG}/model_full.cbm" ]; then
                echo "no user-study selector exp_res/cb_${src}${US_TAG}"; status=1; continue
            fi
            mkdir -p "$out"
            echo "=== user-study selector of $src -> attacks of $model -> $out $(date)"
            CUDA_VISIBLE_DEVICES=${gpus[0]} python scripts/train_pool_catboost.py \
                --apply_model "exp_res/cb_${src}${US_TAG}" --pool_dir "$pool" --out_dir "$out" \
                --alpha_star_from "exp_res/cb_${model}${US_TAG}" "${extra[@]}" 2>&1 | tee "$out/run.log"
            [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "$src -> $model FAILED"; status=1; }
        done
    done
fi

if [[ " $STEPS " == *" table "* ]]; then
    echo "##### 4/4 the analysis table $(date)"
    python scripts/build_paper_table.py --tag "_cs${sfx}" --models $MODELS --transfer_from "$TRANSFER_FROM" \
        --mobile_tag none --alpha_star_from "cb_{model}${US_TAG}" \
        --selectors "cb_cs=cb_{model}_cs${sfx}" "cb_us=cb_{model}${US_TAG}_on_cs${sfx}" \
                    "cbT_us=cb_${TRANSFER_FROM}${US_TAG}_on_{model}_cs${sfx}" \
                    "cbT_cs=cb_${TRANSFER_FROM}_on_{model}_cs${sfx}" \
        --out "exp_res/paper_table_cs${sfx}" || status=1
fi

echo "##### done $(date), status $status (log: $LOG)"
exit $status
