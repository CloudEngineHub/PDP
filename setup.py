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
    ]
)