CONFIG="configs/stable-diffusion/v2-finetune_multi_synthetic.yaml"
CKPT_PATH="your_initial_checkpoint_path/v2-1_512-ema-pruned.ckpt"
LOGDIR="/DreamCache/logs/"
SCRIPT="main.py"


python $SCRIPT \
--base $CONFIG -t \
--actual_resume $CKPT_PATH \
--logdir $LOGDIR \
--memory_path "" \
-n job_name \
--gpus "0,1,2,3" \
--data_root "/path_to_dataset" \