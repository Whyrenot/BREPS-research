#!/usr/bin/env bash
# The paper's candidate pool for SAM-1, SAM-2.1 and SAM-3 (scripts/build_candidate_pool.py
# -> exp_res/pool_<model>/pool.csv.gz, per_box.csv.gz, rows.csv.gz): undef,
# head-select, grad 30 steps with and without multimask, BoN with centre noise
# with and without multimask, steering at L2.norm3 at every alpha with and
# without multimask. Each model needs its finished scan (exp_res/steer_scan_<model>:
# the steering vector). SAM-3 re-execs itself into the sam3 conda env.
#
# Models run one after the other; each model's images are split over the GPUs
# of GPUS, one process per GPU (shard_<i>/), merged on the CPU afterwards.
# A failing model does not stop the next.
#
#   GPUS="0 1 2 3 4 5 6 7" bash scripts/run_candidate_pool.sh
#   MODELS="SAM3" ...   only some models;  SMOKE=1 ...  2 images per half / subset, out dirs *_smoke
#   THREADS=4 ...       CPU threads per process (the mask features run on the CPU)
#   USE=mp ...          the mobile and the desktop annotations (default m: mobile only); the out
#                       dirs get TAG (default _mp), so the mobile pools are left alone
#   CS_DIR=exp_res/critical_shifts_full ...   the model's attacks instead of the user annotations:
#                       <CS_DIR>/<model>/critical_shifts_all.json (sample_critical_shifts.py), TAG default _cs
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
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}
GPUS=${GPUS//,/ }                 # "0,1,2" or "0 1 2"
THREADS=${THREADS:-4}
SMOKE=${SMOKE:-0}
USE=${USE:-m}
CS_DIR=${CS_DIR:-}
if [ -n "$CS_DIR" ]; then TAG=${TAG-_cs}; else TAG=${TAG-$([ "$USE" = "m" ] && echo "" || echo "_$USE")}; fi
status=0

extra=()
suffix=""
if [ "$SMOKE" = "1" ]; then
    extra=(--limit_images 2)
    suffix="_smoke"
fi
read -r -a gpus <<< "$GPUS"
n=${#gpus[@]}

for model in $MODELS; do
    case "$model" in
        SAM)    ckpt=$CKPT_SAM;   ref=results/samfull_m.csv ;;
        SAM2.1) ckpt=$CKPT_SAM21; ref=$REF_SAM21 ;;
        SAM3)   ckpt=$CKPT_SAM3;  ref=results/sam3full_m.csv ;;
        *)      echo "unknown model $model"; status=1; continue ;;
    esac
    scan=exp_res/steer_scan_${model}
    if [ ! -f "$scan/vectors.npz" ]; then echo "$model: no scan in $scan"; status=1; continue; fi
    cs=()
    if [ -n "$CS_DIR" ]; then
        js=$CS_DIR/$model/critical_shifts_all.json
        if [ ! -f "$js" ]; then echo "$model: no $js"; status=1; continue; fi
        cs=(--critical_shifts "$js"); ref=""          # the user-study references do not apply
    fi
    out=exp_res/pool_${model}${TAG}${suffix}
    mkdir -p "$out"
    echo "=== $model pool (--use $USE${CS_DIR:+, attacks of $CS_DIR}) on GPUs ${gpus[*]} -> $out $(date)"
    pids=()
    for i in "${!gpus[@]}"; do
        mkdir -p "$out/shard_$i"
        CUDA_VISIBLE_DEVICES=${gpus[$i]} OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
            python scripts/build_candidate_pool.py --model_name "$model" --root "$ROOT" --checkpoint_path "$ckpt" \
            --scan_dir "$scan" --site "$SITE" --use "$USE" --reference_csv "$ref" --shard "$i" --n_shards "$n" \
            --out_dir "$out/shard_$i" "${cs[@]}" "${extra[@]}" > "$out/shard_$i/run.log" 2>&1 &
        pids+=($!)
    done
    failed=0
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "$model shard $i FAILED (see $out/shard_$i/run.log)"; failed=1; }
    done
    if [ "$failed" -ne 0 ]; then status=1; continue; fi
    echo "=== $model merge $(date)"
    python scripts/build_candidate_pool.py --merge --out_dir "$out" 2>&1 | tee "$out/merge.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "$model merge FAILED"; status=1; }
done

echo "=== done $(date), status $status"
exit $status
