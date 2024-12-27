# PDP

## Setup

TODO: download dataset, setup environment

## Training a Policy

```python
python train.py \
    --config-name=bumpem.yaml \
    hydra.run.dir='outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${logging.name}' \
    logging.name=bumpem \
    dataset.zarr_path=data/bumpem_dass-noise-level-0.12 \
```


# TODO:
- Remove optimizer configuration from policy and transformer class. This should be done in workspace.