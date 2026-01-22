#!/usr/bin/env zsh

SESSION_NAME=${1:-my_session}

COMMANDS=(
    #sam1
    'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_boxes_model_sam.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM/sam_vit_b_01ec64.pth --deterministic  --save-ious --datasets=ADE20K,PascalVOC  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5 --optim_min --modality=bbox --lambda_mult 0.1'
    'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=1 python3 scripts/evaluate_boxes_model_sam.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM/sam_vit_h_4b8939.pth --deterministic  --save-ious --datasets=ADE20K,PascalVOC  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5 --optim_min --modality=bbox --lambda_mult 0.1'
    'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=2 python3 scripts/evaluate_boxes_model_sam.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM/sam_vit_l_0b3195.pth --deterministic  --save-ious --datasets=ADE20K,PascalVOC  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5 --optim_min --modality=bbox --lambda_mult 0.1'

    # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=3 python3 scripts/evaluate_boxes_model_sam.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM/sam_vit_b_01ec64.pth  --deterministic  --save-ious --datasets=ADE20K,PascalVOC     --n_opt_steps=50 --lr_mult=9  --iou-analysis --gpus="0" --thresh=0.5   --lambda_mult 0.1 --modality=bbox'
    # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=4 python3 scripts/evaluate_boxes_model_sam.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM/sam_vit_h_4b8939.pth  --deterministic  --save-ious --datasets=ADE20K,PascalVOC     --n_opt_steps=50 --lr_mult=9  --iou-analysis --gpus="0" --thresh=0.5   --lambda_mult 0.1 --modality=bbox'
    # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=5 python3 scripts/evaluate_boxes_model_sam.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM/sam_vit_l_0b3195.pth  --deterministic  --save-ious --datasets=ADE20K,PascalVOC     --n_opt_steps=50 --lr_mult=9  --iou-analysis --gpus="0" --thresh=0.5   --lambda_mult 0.1 --modality=bbox'

    #robust-sam

    # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=1 python3 scripts/evaluate_boxes_model_robustsam.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/RobustSAM/robustsam_checkpoint_b.pth --deterministic  --save-ious --datasets=Berkeley,GrabCut,COCO_MVal,DAVIS,ADE20K,PascalVOC  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5 --optim_min --modality=bbox --lambda_mult 0.1'
    # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=1 python3 scripts/evaluate_boxes_model_robustsam.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/RobustSAM/robustsam_checkpoint_h.pth --deterministic  --save-ious --datasets=Berkeley,GrabCut,COCO_MVal,DAVIS,ADE20K,PascalVOC  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5 --optim_min --modality=bbox --lambda_mult 0.1'
    # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=2 python3 scripts/evaluate_boxes_model_robustsam.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/RobustSAM/robustsam_checkpoint_l.pth --deterministic  --save-ious --datasets=Berkeley,GrabCut,COCO_MVal,DAVIS,ADE20K,PascalVOC  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5 --optim_min --modality=bbox --lambda_mult 0.1'

    # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=3 python3 scripts/evaluate_boxes_model_robustsam.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/RobustSAM/robustsam_checkpoint_b.pth  --deterministic  --save-ious --datasets=Berkeley,GrabCut,COCO_MVal,DAVIS,ADE20K,PascalVOC     --n_opt_steps=50 --lr_mult=9  --iou-analysis --gpus="0" --thresh=0.5   --lambda_mult 0.1 --modality=bbox'
    # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=4 python3 scripts/evaluate_boxes_model_robustsam.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/RobustSAM/robustsam_checkpoint_h.pth  --deterministic  --save-ious --datasets=Berkeley,GrabCut,COCO_MVal,DAVIS,ADE20K,PascalVOC     --n_opt_steps=50 --lr_mult=9  --iou-analysis --gpus="0" --thresh=0.5   --lambda_mult 0.1 --modality=bbox'
    # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=5 python3 scripts/evaluate_boxes_model_robustsam.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/RobustSAM/robustsam_checkpoint_l.pth  --deterministic  --save-ious --datasets=Berkeley,GrabCut,COCO_MVal,DAVIS,ADE20K,PascalVOC     --n_opt_steps=50 --lr_mult=9  --iou-analysis --gpus="0" --thresh=0.5   --lambda_mult 0.1 --modality=bbox'

    #sam2.1-hq
   'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=6 python3 scripts/evaluate_boxes_model_sam2.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM2-HQ/sam2.1_hq_hiera_large.pt  --deterministic  --save-ious --datasets=TETRIS  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5 --optim_min --modality=bbox --lambda_mult 0.1'
   'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_boxes_model_sam2.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM2-HQ/sam2.1_hq_hiera_large.pt  --deterministic  --save-ious --datasets=TETRIS  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5  --modality=bbox --lambda_mult 0.1'
   #
    #sam2.1
   # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=6 python3 scripts/evaluate_boxes_model_sam2.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM2.1/sam2.1_hiera_base_plus.pt  --deterministic  --save-ious --datasets=ADE20K,PascalVOC  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5 --optim_min --modality=bbox --lambda_mult 0.1'
   # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=7 python3 scripts/evaluate_boxes_model_sam2.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM2.1/sam2.1_hiera_small.pt  --deterministic  --save-ious --datasets=ADE20K,PascalVOC  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5 --optim_min --modality=bbox --lambda_mult 0.1'
   # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_boxes_model_sam2.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM2.1/sam2.1_hiera_large.pt  --deterministic  --save-ious --datasets=ADE20K,PascalVOC  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5 --optim_min --modality=bbox --lambda_mult 0.1'

   # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=1 python3 scripts/evaluate_boxes_model_sam2.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM2.1/sam2.1_hiera_base_plus.pt  --deterministic  --save-ious --datasets=ADE20K,PascalVOC  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5  --modality=bbox --lambda_mult 0.1'
   # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=2 python3 scripts/evaluate_boxes_model_sam2.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM2.1/sam2.1_hiera_small.pt  --deterministic  --save-ious --datasets=ADE20K,PascalVOC  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5  --modality=bbox --lambda_mult 0.1'
   # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=3 python3 scripts/evaluate_boxes_model_sam2.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM2.1/sam2.1_hiera_large.pt  --deterministic  --save-ious --datasets=ADE20K,PascalVOC  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5  --modality=bbox --lambda_mult 0.1'

   #sam-hq
   # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=4 python3 scripts/evaluate_boxes_model_samhq.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM-HQ/sam_hq_vit_b.pth --deterministic  --save-ious --datasets=ADE20K,PascalVOC  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5 --optim_min --modality=bbox --lambda_mult 0.1'
   # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=5 python3 scripts/evaluate_boxes_model_samhq.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM-HQ/sam_hq_vit_h.pth --deterministic  --save-ious --datasets=ADE20K,PascalVOC  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5 --optim_min --modality=bbox --lambda_mult 0.1'
   # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=6 python3 scripts/evaluate_boxes_model_samhq.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM-HQ/sam_hq_vit_l.pth --deterministic  --save-ious --datasets=ADE20K,PascalVOC  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5 --optim_min --modality=bbox --lambda_mult 0.1'

   # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=7 python3 scripts/evaluate_boxes_model_samhq.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM-HQ/sam_hq_vit_b.pth --deterministic  --save-ious --datasets=ADE20K,PascalVOC  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5 --modality=bbox --lambda_mult 0.1'
   # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_boxes_model_samhq.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM-HQ/sam_hq_vit_h.pth --deterministic  --save-ious --datasets=ADE20K,PascalVOC  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5 --modality=bbox --lambda_mult 0.1'
   # 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=1 python3 scripts/evaluate_boxes_model_samhq.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM-HQ/sam_hq_vit_l.pth --deterministic  --save-ious --datasets=ADE20K,PascalVOC  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5 --modality=bbox --lambda_mult 0.1'
   #mobile sam
# 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=2 python3 scripts/evaluate_boxes_model_mobilesam.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/MobileSAM/mobile_sam.pt --deterministic  --save-ious --datasets=ADE20K,PascalVOC  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5 --optim_min --modality=bbox --lambda_mult 0.1'
# 'cd BREPS && conda activate ../.conda && CUDA_VISIBLE_DEVICES=3 python3 scripts/evaluate_boxes_model_mobilesam.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/MobileSAM/mobile_sam.pt --deterministic  --save-ious --datasets=ADE20K,PascalVOC  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5 --modality=bbox --lambda_mult 0.1'
)

NAMES=(
  # sambench-b-min
  # sambench-h-min
  # sambench-l-min

  # sambench-b-max
  # sambench-h-max
  # sambench-l-max

  # rsambench-b-min
  # rsambench-h-min
  # rsambench-l-min

  # rsambench-b-max
  # rsambench-h-max
  # rsambench-l-max

  samhq2bench-l-min
  samhq2bench-l-max

  # sam2bench-b-min
  # sam2bench-s-min
  # sam2bench-l-min

  # sam2bench-b-max
  # sam2bench-s-max
  # sam2bench-l-max


  # samhqbench-b-min
  # samhqbench-h-min
  # samgqbench-l-min

  # samhqbench-b-max
  # samhqbench-h-max
  # samhqbench-l-max

  # mobilesam-max
  # mobilesam-l-max
)

if (( ${#COMMANDS} != ${#NAMES} )); then
  echo "Error: COMMANDS and NAMES must have the same length."
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME"

total=${#COMMANDS}
for (( i=1; i<=total; i++ )); do
  cmd=${COMMANDS[i]}
  name=${NAMES[i]}
  tmux_index=$((i-1))

  if (( i == 1 )); then
    tmux rename-window -t "$SESSION_NAME:0" "$name"
    tmux send-keys     -t "$SESSION_NAME:0" "$cmd" C-m
  else
    tmux new-window    -t "$SESSION_NAME" -n "$name"
    tmux send-keys     -t "$SESSION_NAME:$tmux_index" "$cmd" C-m
  fi
done

echo "Attach with: tmux attach -t $SESSION_NAME"
