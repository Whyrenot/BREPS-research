#!/usr/bin/env bash
# One steering vector for every subset? scripts/steer_universal.py for SAM-1, SAM-2.1 and SAM-3:
# the GPU pass (every subset's vector from its half A; own / COCO / pooled / leave-one-out /
# random vectors on every half), then the tables on the CPU (--from_dump), per model:
#   exp_res/steer_universal/<model>_<site>[_<use>]/rows.csv.gz, vectors.npz, summary.md, table.csv, pairs.csv, ...
# Each model needs its finished scan (exp_res/steer_scan_<model>: the COCO A vector, checked).
# The models are spread over GPUS, one process per GPU (models that share a GPU run one after the
# other); SAM-3 re-execs itself into the sam3 conda env. A failing model does not stop the others.
#
#   GPUS="0 1 2" bash scripts/run_steer_universal.sh
#   MODELS="SAM3" ...      only some models
#   SITE=L2.norm2 ...      another token site (default L2.norm3, the pool's)
#   USE=mp ...             the mobile and the desktop annotations (default m, as the scans)
#   SMOKE=1 ...            2 images per half and subset, alphas 4 and 16, out dirs *_smoke
#   STEPS=tables ...       only the CPU tables from an existing rows.csv.gz (default "gpu tables")
set -uo pipefail
cd "$(dirname "$0")/.."

ROOT=${ROOT:-/home/jovyan/shares/SR006.nfs2/pishugin/user_study/FOR_TEST}
CKPTS=${CKPTS:-/home/jovyan/shares/SR006.nfs2/pishugin/temp/rclicks/MODEL_CHECKPOINTS}
CKPT_SAM=${CKPT_SAM:-$CKPTS/SAM/sam_vit_b_01ec64.pth}
CKPT_SAM21=${CKPT_SAM21:-$CKPTS/SAM2.1/sam2.1_hiera_base_plus.pt}
CKPT_SAM3=${CKPT_SAM3:-}          # "" = the builder reuses facebook/sam3 from the HF cache
REF_SAM21=${REF_SAM21:-}          # results/sam21full_m.csv is hiera_large: no reference for base_plus
MODELS=${MODELS:-"SAM SAM2.1 SAM3"}
SITE=${SITE:-L2.norm3}
USE=${USE:-m}
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}
GPUS=${GPUS//,/ }                 # "0,1,2" or "0 1 2"
THREADS=${THREADS:-8}
STEPS=${STEPS:-"gpu tables"}
SMOKE=${SMOKE:-0}

extra=()
suffix=""
[ "$USE" = "m" ] || suffix="_$USE"
if [ "$SMOKE" = "1" ]; then
    extra=(--limit_images 2 --alphas 4,16)
    suffix="${suffix}_smoke"
fi
read -r -a gpus <<< "$GPUS"
read -r -a models <<< "$MODELS"
n=${#gpus[@]}
mkdir -p exp_res/steer_universal
LOG=exp_res/steer_universal/run_${SITE}${suffix}.log
exec > >(tee -a "$LOG") 2>&1
echo "##### run_steer_universal: MODELS=$MODELS SITE=$SITE USE=$USE GPUS=${gpus[*]} STEPS=$STEPS SMOKE=$SMOKE $(date)"

run_model() {
    local model=$1 gpu=$2 ckpt ref
    case "$model" in
        SAM)    ckpt=$CKPT_SAM;   ref=results/samfull_m.csv ;;
        SAM2.1) ckpt=$CKPT_SAM21; ref=$REF_SAM21 ;;
        SAM3)   ckpt=$CKPT_SAM3;  ref=results/sam3full_m.csv ;;
        *)      echo "unknown model $model"; return 1 ;;
    esac
    local scan=exp_res/steer_scan_${model}
    local out=exp_res/steer_universal/${model}_${SITE}${suffix}
    mkdir -p "$out"
    if [[ " $STEPS " == *" gpu "* ]]; then
        [ -f "$scan/vectors.npz" ] || { echo "$model: no scan in $scan"; return 1; }
        echo "=== $model GPU pass on GPU $gpu -> $out (log $out/run.log) $(date)"
        CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
            python scripts/steer_universal.py --model_name "$model" --root "$ROOT" --checkpoint_path "$ckpt" \
            --scan_dir "$scan" --site "$SITE" --use "$USE" --reference_csv "$ref" --out_dir "$out" "${extra[@]}" \
            > "$out/run.log" 2>&1 || { echo "$model GPU pass FAILED (see $out/run.log)"; return 1; }
        tail -n 2 "$out/run.log"
    fi
    if [[ " $STEPS " == *" tables "* ]]; then
        echo "=== $model tables $(date)"
        OMP_NUM_THREADS=$THREADS python scripts/steer_universal.py --from_dump "$out" > "$out/tables.log" 2>&1 \
            || { echo "$model tables FAILED (see $out/tables.log)"; return 1; }
        echo "=== $model done: $out/summary.md $(date)"
    fi
}

pids=()
for g in "${!gpus[@]}"; do
    (
        st=0
        for j in "${!models[@]}"; do
            [ $((j % n)) -eq "$g" ] || continue
            run_model "${models[$j]}" "${gpus[$g]}" || st=1
        done
        exit $st
    ) &
    pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "##### done $(date), status $status (log: $LOG)"
exit $status
