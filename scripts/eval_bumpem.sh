ckpt_path="outputs/2024.12.30/03.24.01_bumpem/checkpoints/checkpoint_epoch_1500.ckpt" # your ckpt path

python pdp/bumpem/evaluate.py \
    --ckpt_path ${ckpt_path} \
    --save_video \