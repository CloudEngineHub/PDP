# HYDRA_FULL_ERROR=1 python train.py \
#     --config-name=bumpem.yaml \
#     hydra.run.dir=outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${logging.name} \
#     logging.name=bumpem \
#     dataset.zarr_path=data/bumpem_dass-noise-level-0.12 \

HYDRA_FULL_ERROR=1 python train.py \
    --config-name=bumpem.yaml \
    hydra.run.dir=outputs/debug \
    logging.name=bumpem \
    dataset.zarr_path=data/bumpem_dass-noise-level-0.12/data.zarr \
    training.debug=True training.logging=False \