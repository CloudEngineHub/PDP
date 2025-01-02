ckpt_path="" # your ckpt path

python pdp/bumpem/evaluate.py \
    --ckpt_path ${ckpt_path} \
    --save_video \


### Example of modifying the perturbation parameters
# python pdp/bumpem/evaluate.py \
#     --ckpt_path ${ckpt_path} \
#     --save_video \
#     --p_frc_frac 0.2 \
#     --p_ang 270 \
#     --imp_time 0.5 \