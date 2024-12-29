# PDP

## Setup

Placeholder

## Training a Policy

```python
python train.py \
    --config-name=bumpem.yaml \
    hydra.run.dir='outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${logging.name}' \
    logging.name=bumpem \
    dataset.zarr_path=data/bumpem_dass-noise-level-0.12/data.zarr \
```

## Evaluating a Policy

For Bump-em:


## TODO: