#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
CKPT_PATH="/models/ldm/stable_diffusion_v2/v2-1_512-ema-pruned.ckpt"
FT_PATH='/personalized_checkpoint_folder'
OUTDIR_BASE="out_dir"
SCRIPT="scripts/memory_text2img_multi_ref.py"


python $SCRIPT \
    --ddim_eta 1.0 \
    --n_samples 5 \
    --n_iter 1 \
    --scale 7.5 \
    --image_scale 0.3 \
    --ddim_steps 50 \
    --ckpt_path $CKPT_PATH \
    --ft_path $FT_PATH \
    --load_step 199999 \
    --outdir "${OUTDIR_BASE}/" \
    --prompt "A dragon, painted on a wall." \
    --image_path "path_to_your_image" 
  