#!/bin/bash
export CUDA_VISIBLE_DEVICES="3"
# Define paths and settings
CKPT_PATH="/models/ldm/stable_diffusion2/v2-1_512-ema-pruned.ckpt"
FT_PATH='/pretrain/dreamcache_sd2.1'
OUTDIR_BASE="/evals/SD2.1"
SCRIPT="scripts/memory_text2img_multi_ref.py"

declare -a LIVE_SUBJECT_CLASSES=("cat" "cat2" "dog" "dog2" "dog3" "dog5" "dog6" "dog7" "dog8") 
declare -a OBJECT_CLASSES=("backpack" "backpack_dog" "bear_plushie" "berry_bowl" "can" "candle" "clock" "colorful_sneaker" "fancy_boot" "grey_sloth_plushie" "monster_toy" "pink_sunglasses" "poop_emoji" "rc_car" "red_cartoon" "robot_toy" "shiny_sneaker" "teapot" "vase" "duck_toy" "wolf_plushie")

# Function to run prompts for a given category
for CLASS in "${OBJECT_CLASSES[@]}"
    do
    echo "Running for class: $CLASS with prompt: '$PROMPT'"
        # Run the Python script with the dynamic prompt
        python $SCRIPT \
            --ddim_eta 0.0 \
            --n_samples 4 \
            --n_iter 1 \
            --scale 7.5 \
            --image_scale 0.4 \
            --ddim_steps 50 \
            --ckpt_path $CKPT_PATH \
            --ft_path $FT_PATH \
            --load_step 199999 \
            --from-file /scripts/prompt_objects_dreambooth.txt \
            --image_path /home/CORP/emanuele.aiello/dreambench/dataset/$CLASS/00.png \
            --outdir "${OUTDIR_BASE}/$CLASS" \
            --class_key $CLASS
    done


for CLASS in "${LIVE_SUBJECT_CLASSES[@]}"
    do
    echo "Running for class: $CLASS with prompt: '$PROMPT'"
        # Run the Python script with the dynamic prompt
        python $SCRIPT \
            --ddim_eta 0.0 \
            --n_samples 4 \
            --n_iter 1 \
            --scale 7.5 \
            --image_scale 0.4 \
            --ddim_steps 50 \
            --ckpt_path $CKPT_PATH \
            --ft_path $FT_PATH \
            --load_step 199999 \
            --from-file /scripts/prompts_living_dreambooth.txt \
            --image_path /home/CORP/emanuele.aiello/dreambench/dataset/$CLASS/00.png \
            --outdir "${OUTDIR_BASE}/$CLASS" \
            --class_key $CLASS
    done