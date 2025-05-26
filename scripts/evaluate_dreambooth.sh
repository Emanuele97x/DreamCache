#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"

declare -a OBJECT_CLASSES=("backpack" "backpack_dog" "bear_plushie" "berry_bowl" "can" "candle" "clock" "colorful_sneaker" "fancy_boot" "grey_sloth_plushie" "monster_toy" "pink_sunglasses" "poop_emoji" "rc_car" "red_cartoon" "robot_toy" "shiny_sneaker" "teapot" "vase" "duck_toy" "wolf_plushie" )
declare -a LIVE_SUBJECT_CLASSES=("cat" "cat2" "dog" "dog2" "dog3" "dog5" "dog6" "dog7" "dog8")


for CLASS in "${OBJECT_CLASSES[@]}"
do
    echo "Running evaluation for class: $CLASS"
    python scripts/evaluate_objects_dreambooth.py \
        --data_dir /dreambooth/dataset/$CLASS \
        --generated_dir /evals/SD2.1/$CLASS/samples \
        --class_name $CLASS \
        --tag SD2.1
done

for CLASS in "${LIVE_SUBJECT_CLASSES[@]}"
do
    echo "Running evaluation for class: $CLASS"
    python scripts/evaluate_living_dreambooth.py \
        --data_dir /dreambooth/dataset/$CLASS \
        --generated_dir /evals/SD2.1/$CLASS/samples \
        --class_name $CLASS \
        --tag SD2.1
done 


python scripts/average_scores_dreambooth.py --tag SD2.1