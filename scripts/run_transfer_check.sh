#!/usr/bin/env bash
# "Do our methods hurt on a new set?": the paper's candidate pool on a transfer set of
# fetch_transfer_sets.py, then the user-study CatBoost of the same model applied to it
# (no retraining), every method against the undefended model.
#   1. build_candidate_pool.py --plain_root DATA   -> exp_res/pool_<model>_<NAME>/
#      (images split over GPUS, one process per GPU, merged on the CPU)
#   2. train_pool_catboost.py --apply_model exp_res/cb_<model>_mp --alpha_star_from the same
#      --regroup dataset                         -> exp_res/cb_<model>_mp_on_<NAME>/summary.md
# Steering: the scan's vector of all COCO A (as for every non-A image in the paper),
# alpha* frozen from cb_<model>_mp. The user-study root is only read for the COCO split.
#
#   DATA=/.../datasets/OpenImagesV7_test500 GPUS="0 1 2 3" bash scripts/run_transfer_check.sh
#   MODEL=SAM (default) | SAM2.1 | SAM3;  NAME=OpenImages;  SMOKE=1: 4 images, dirs *_smoke
#   STEPS="cb" ...  only the CatBoost step (the pool is there)
set -uo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:?set DATA to a fetch_transfer_sets.py output dir}
NAME=${NAME:-OpenImages}
MODEL=${MODEL:-SAM}
ROOT=${ROOT:-/home/jovyan/shares/SR006.nfs2/pishugin/user_study/FOR_TEST}
CKPTS=${CKPTS:-/home/jovyan/shares/SR006.nfs2/pishugin/temp/rclicks/MODEL_CHECKPOINTS}
case "$MODEL" in
    SAM)    CKPT=${CKPT:-$CKPTS/SAM/sam_vit_b_01ec64.pth} ;;
    SAM2.1) CKPT=${CKPT:-$CKPTS/SAM2.1/sam2.1_hiera_base_plus.pt} ;;
    SAM3)   CKPT=${CKPT:-} ;;
    *) echo "unknown MODEL $MODEL"; exit 1 ;;
esac
SITE=${SITE:-L2.norm3}
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}
GPUS=${GPUS//,/ }
THREADS=${THREADS:-4}
STEPS=${STEPS:-"pool cb"}
SMOKE=${SMOKE:-0}
CB_SRC=${CB_SRC:-exp_res/cb_${MODEL}_mp}
read -r -a gpus <<< "$GPUS"
n=${#gpus[@]}
extra=(); suffix=""
if [ "$SMOKE" = "1" ]; then extra=(--limit_images 4); suffix="_smoke"; fi
POOL=exp_res/pool_${MODEL}_${NAME}${suffix}
OUT=exp_res/cb_${MODEL}_mp_on_${NAME}${suffix}
scan=exp_res/steer_scan_${MODEL}
for f in "$DATA/annotations.csv" "$scan/vectors.npz" "$CB_SRC/model_full.cbm" "$CB_SRC/meta.json"; do
    [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
mkdir -p "$POOL"
exec > >(tee -a "$POOL/transfer_check.log") 2>&1
echo "##### run_transfer_check: $MODEL on $DATA as $NAME, GPUs ${gpus[*]}, CatBoost $CB_SRC $(date)"

if [[ " $STEPS " == *" pool "* ]]; then
    pids=()
    for i in "${!gpus[@]}"; do
        mkdir -p "$POOL/shard_$i"
        CUDA_VISIBLE_DEVICES=${gpus[$i]} OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
            python scripts/build_candidate_pool.py --model_name "$MODEL" --root "$ROOT" --checkpoint_path "$CKPT" \
            --scan_dir "$scan" --site "$SITE" --use mp --plain_root "$DATA" --plain_name "$NAME" \
            --shard "$i" --n_shards "$n" --out_dir "$POOL/shard_$i" "${extra[@]}" > "$POOL/shard_$i/run.log" 2>&1 &
        pids+=($!)
    done
    trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "shard $i FAILED (see $POOL/shard_$i/run.log)"; exit 1; }
    done
    python scripts/build_candidate_pool.py --merge --out_dir "$POOL" --reference_csv "" || { echo "merge FAILED"; exit 1; }
fi

if [[ " $STEPS " == *" cb "* ]]; then
    mkdir -p "$OUT"
    CUDA_VISIBLE_DEVICES=${gpus[0]} python scripts/train_pool_catboost.py --apply_model "$CB_SRC" --apply_with full \
        --alpha_star_from "$CB_SRC" --regroup dataset --pool_dir "$POOL" --out_dir "$OUT" --devices 0 \
        || { echo "CatBoost FAILED"; exit 1; }
    echo "##### done $(date): $OUT/summary.md"
fi
