from setuptools import setup


setup(
    name='pdp',
    version='0.1',
    packages=['pdp'],
    install_requires=[
        'torch==1.12.1',
        'torchvision==0.13.1',
        'numpy==1.23.3',
        'numba==0.56.4',
        'zarr==2.12.0',
        'hydra-core==1.2.0',
        'mujoco==3.2.3',
        'wandb==0.19.1',
        'einops==0.8.0',
        'tqdm==4.67.1',
        'diffusers==0.32.1',
        'transformers==4.46.3',
    ]
)