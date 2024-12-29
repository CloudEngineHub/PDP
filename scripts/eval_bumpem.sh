ckpt_path="outputs/2024.12.29/01.24.31_bumpem/checkpoints/checkpoint_epoch_1500.ckpt" # your ckpt path

python pdp/bumpem/evaluate.py \
    --ckpt_path ${ckpt_path} \
    --save_video \