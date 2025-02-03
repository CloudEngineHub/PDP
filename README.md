# PDP Official Codebase

[![Paper](https://img.shields.io/badge/Paper-blue)](https://dl.acm.org/doi/full/10.1145/3680528.3687683)
[![Project Site](https://img.shields.io/badge/Project%20Site-grey.svg)](https://stanford-tml.github.io/PDP.github.io/)
[![The Movement Lab](https://img.shields.io/badge/The%20Movement%20Lab-red.svg)](https://tml.stanford.edu/)

Official codebase for PDP. This codebase currently supports the perturbation recovery task from the PDP paper. Our code is based heavily on the <a href="https://github.com/real-stanford/diffusion_policy" target="_blank">Diffusion Policy</a> codebase. We take only the necessary parts of their codebase to simplify the PDP implementation.


## Setup

Clone the repo and install the dependencies. Modify the pytorch installation to fit your system.

```bash
git clone https://github.com/Stanford-TML/PDP.git
cd PDP

conda create -n pdp python=3.8
conda activate pdp
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -e .
```

Then download the Bump-em dataset from <a href="https://drive.google.com/drive/folders/1HQFb6oLCYiveConeTb-Q9FZBMmX0wREg?usp=drive_link" target="_blank">here</a> and put it in the `data/` directory and extract it.

```bash
tar -xzvf /path/to/dataset.tar.gz -C data/
```


## Training a Policy

To train a policy, run the following script. Parameters defined in `cfg/bumpem.yaml` can be modified from the command line via hydra.

```bash
bash scripts/train_bumpem.sh
```


## Running a Policy

To evaluate a policy visually, run the following script.

```bash
bash scripts/eval_bumpem.sh
```

The result should look something like this:

<img src="assets/bumpem_eval_result.gif" alt="" width="256" height="256" style="border-radius: 5px;">

The parameters of the perturbation can be modified from the command line. The parameters that can be modified are: the magnitude of the force (as a percentage of the skeleton's body weight), the time the perturbation begins, and the direction of the perturbation. An example is given in `scripts/eval_bumpem.sh`.

## Additional Details

### Bump-em Dataset

The dataset contains observation and action data collected on the perturbation recovery task. The observation is composed of the following quantities for 10 of the bodies in the model, plus a perturbation signal, yielding a 181 dimensional observation:
- 3D body position (3 * 10 values)
- 3x3 rotation matrix (9 * 10 values)
- 6D linear + angular velocities (6 * 10 values)
- Perturbation signal (1 value)

The actions are the 25 DoF desired joint positions. The data was collected at 50Hz.