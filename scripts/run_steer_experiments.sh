#!/usr/bin/env bash
# The four steering experiments of 2026-10-27, one after the other, on one GPU:
#   1-3  scripts/steer_variants.py         -> exp_res/steer_variants/summary.md
#   4    scripts/dump_steer_candidates.py  -> results/sam1_steer_candidates_m.csv
#        (then, on the laptop: python scripts/selector_steer.py)
# A failing step does not stop the next one; the exit code says whether any failed.
#
#   CUDA_VISIBLE_DEVICES=3 bash scripts/run_steer_experiments.sh
set -uo pipefail
cd "$(dirname "$0")/.."

ROOT=${ROOT:-/home/jovyan/shares/SR006.nfs2/pishugin/user_study/FOR_TEST}
CKPT=${CKPT:-/home/jovyan/shares/SR006.nfs2/pishugin/temp/rclicks/MODEL_CHECKPOINTS/SAM/sam_vit_b_01ec64.pth}
VEC=${VEC:-exp_res/steer_user_study/vector.npz}
REF=${REF:-results/samfull_m.csv}
mkdir -p exp_res/steer_variants results
status=0

echo "=== [1-3] steer_variants $(date)"
python scripts/steer_variants.py --root "$ROOT" --checkpoint_path "$CKPT" --vector "$VEC" \
    --reference_csv "$REF" --out_dir exp_res/steer_variants 2>&1 | tee exp_res/steer_variants/run.log
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "steer_variants FAILED"; status=1; }

echo "=== [4] dump_steer_candidates $(date)"
python scripts/dump_steer_candidates.py --root "$ROOT" --checkpoint_path "$CKPT" --vector "$VEC" \
    --out_csv results/sam1_steer_candidates_m.csv 2>&1 | tee results/sam1_steer_candidates_m.log
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "dump_steer_candidates FAILED"; status=1; }

echo "=== done $(date), status $status"
exit $status
