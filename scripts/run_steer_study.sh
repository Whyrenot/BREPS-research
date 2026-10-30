#!/usr/bin/env bash
# The full steering study (scripts/steer_study.py) for SAM-1, SAM-2.1 and SAM-3 on the
# m + p annotations: D10 - D1 vectors per subset at L2.norm2, L2.norm3 and the hypernetwork
# outputs; own / pooled / leave-one-dataset-out (+ COCO, random) vectors out of fold on 5 image
# folds; every alpha, both output modes; the CatBoost alpha regressor out of fold.
# Per model:  states (all GPUs) -> vectors (CPU) -> steer (all GPUs) -> merge (CPU); then, after the
# GPU stages of every model, the analyses of all models at once, each with CatBoost on its own GPU
# of GPUS (at most --cb_gpu_ram_part 0.25 of the card)
#   exp_res/steer_study/<model>/summary.md, per_box.csv.gz, table.csv, pairs.csv, steer.npz, ...
# merge checks undef / head-select against the model's m + p pool (exp_res/pool_<model>_mp).
# SAM-3 re-execs itself into the sam3 conda env. A failing model does not stop the next.
#
#   GPUS="0 1 2 3 4 5 6 7" bash scripts/run_steer_study.sh
#   MODELS="SAM3" ...        only some models
#   STEPS="analyse" ...      only some steps of: states vectors steer merge analyse
#   SMOKE=1 ...              2 images per subset, alphas 4 and 16 (hyper 4 and 32), CatBoost 50 iterations, *_smoke
#   CB_TASK=CPU ...          the alpha regressor on the CPU (default GPU)
#   SITES=... VARIANTS=... RULES=... the study's grid (default L2.norm2,L2.norm3,hyper /
#                            own,pool,lodo,coco,rand / decile,halves: D10 - D1 and 13 best - 12 worst)
# GPUS: free cards only. The states step clears the model's old shard_* (a stale shard of a run
# with more GPUs would mix in). STEPS="analyse" MODELS=... reruns the analysis alone.
set -uo pipefail
cd "$(dirname "$0")/.."

ROOT=${ROOT:-/home/jovyan/shares/SR006.nfs2/pishugin/user_study/FOR_TEST}
CKPTS=${CKPTS:-/home/jovyan/shares/SR006.nfs2/pishugin/temp/rclicks/MODEL_CHECKPOINTS}
CKPT_SAM=${CKPT_SAM:-$CKPTS/SAM/sam_vit_b_01ec64.pth}
CKPT_SAM21=${CKPT_SAM21:-$CKPTS/SAM2.1/sam2.1_hiera_base_plus.pt}
CKPT_SAM3=${CKPT_SAM3:-}          # "" = the builder reuses facebook/sam3 from the HF cache
MODELS=${MODELS:-"SAM SAM2.1 SAM3"}
USE=${USE:-mp}
SITES=${SITES:-L2.norm2,L2.norm3,hyper}
VARIANTS=${VARIANTS:-own,pool,lodo,coco,rand}
RULES=${RULES:-decile,halves}
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}
GPUS=${GPUS//,/ }
THREADS=${THREADS:-4}
CB_TASK=${CB_TASK:-GPU}
STEPS=${STEPS:-"states vectors steer merge analyse"}
SMOKE=${SMOKE:-0}

study=(--use "$USE" --sites "$SITES" --variants "$VARIANTS" --rules "$RULES")
analyse=(--task_type "$CB_TASK")
suffix=""
if [ "$SMOKE" = "1" ]; then
    study+=(--limit_images 2 --alphas 4,16 --alphas_hyper 4,32)
    analyse+=(--cb_iterations 50 --n_boot 200 --n_split 20)
    suffix="_smoke"
fi
read -r -a gpus <<< "$GPUS"
n=${#gpus[@]}
mkdir -p exp_res/steer_study
LOG=exp_res/steer_study/run${suffix}.log
exec > >(tee -a "$LOG") 2>&1
# Ctrl-C / kill: background jobs of a non-interactive bash ignore SIGINT, so the shards (and SAM-3's
# conda-run children) would go on; stop the whole process group of this run instead
trap 'trap - INT TERM; echo "##### interrupted $(date): stopping every process of this run"; kill -TERM 0' INT TERM
echo "##### run_steer_study: MODELS=$MODELS USE=$USE SITES=$SITES VARIANTS=$VARIANTS RULES=$RULES GPUS=${gpus[*]} STEPS=$STEPS SMOKE=$SMOKE $(date)"

has() { [[ " $STEPS " == *" $1 "* ]]; }

# one GPU stage of one model, one process per GPU; returns non-zero if any shard failed
shards() {
    local stage=$1 model=$2 ckpt=$3 out=$4 pids=() fail=0
    for i in "${!gpus[@]}"; do
        mkdir -p "$out/shard_$i"
        CUDA_VISIBLE_DEVICES=${gpus[$i]} OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
            python scripts/steer_study.py --stage "$stage" --model_name "$model" --root "$ROOT" \
            --checkpoint_path "$ckpt" --shard "$i" --n_shards "$n" --out_dir "$out" "${study[@]}" \
            > "$out/shard_$i/$stage.log" 2>&1 &
        pids+=($!)
    done
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "$model $stage shard $i FAILED (see $out/shard_$i/$stage.log)"; fail=1; }
    done
    return $fail
}

status=0
ready=()
for model in $MODELS; do
    case "$model" in
        SAM)    ckpt=$CKPT_SAM ;;
        SAM2.1) ckpt=$CKPT_SAM21 ;;
        SAM3)   ckpt=$CKPT_SAM3 ;;
        *)      echo "unknown model $model"; status=1; continue ;;
    esac
    out=exp_res/steer_study/${model}${suffix}
    pool=exp_res/pool_${model}_${USE}
    mkdir -p "$out"
    if has states; then
        echo "=== $model states on GPUs ${gpus[*]} $(date)"
        rm -rf "$out"/shard_*
        shards states "$model" "$ckpt" "$out" || { status=1; continue; }
    fi
    if has vectors; then
        echo "=== $model vectors $(date)"
        python scripts/steer_study.py --stage vectors --out_dir "$out" 2>&1 | tee "$out/vectors.log"
        [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "$model vectors FAILED"; status=1; continue; }
    fi
    if has steer; then
        echo "=== $model steer on GPUs ${gpus[*]} $(date)"
        shards steer "$model" "$ckpt" "$out" || { status=1; continue; }
    fi
    if has merge; then
        echo "=== $model merge $(date)"
        python scripts/steer_study.py --stage merge --out_dir "$out" --pool_dir "$pool" 2>&1 | tee "$out/merge.log"
        [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "$model merge FAILED"; status=1; continue; }
    fi
    ready+=("$model")
done

# the analyses only now, when no shard holds a card: CatBoost of one model per GPU, all at once
if has analyse; then
    bg=()
    for j in "${!ready[@]}"; do
        model=${ready[$j]}
        out=exp_res/steer_study/${model}${suffix}
        gpu=${gpus[$((j % n))]}
        echo "=== $model analyse (CatBoost on $CB_TASK, GPU $gpu; log $out/analyse.log) $(date)"
        ( CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$THREADS python scripts/steer_study.py --stage analyse \
            --out_dir "$out" --devices 0 "${analyse[@]}" > "$out/analyse.log" 2>&1 \
            && echo "=== $model done: $out/summary.md $(date)" || { echo "$model analyse FAILED (see $out/analyse.log)"; exit 1; } ) &
        bg+=($!)
    done
    for p in ${bg[@]+"${bg[@]}"}; do wait "$p" || status=1; done
fi
echo "##### done $(date), status $status (log: $LOG)"
exit $status
