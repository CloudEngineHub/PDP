# PDP

[![Paper](https://img.shields.io/badge/Paper-blue)](https://dl.acm.org/doi/full/10.1145/3680528.3687683)
[![Project Site](https://img.shields.io/badge/Project%20Site-grey.svg)](https://stanford-tml.github.io/PDP.github.io/)
[![The Movement Lab](https://img.shields.io/badge/The%20Movement%20Lab-red.svg)](https://tml.stanford.edu/)

Official codebase for PDP. This codebase currently supports the perturbatino recovery task from the PDP paper.


## Setup

Clone the repo and install the dependencies.

```bash
git clone https://github.com/Stanford-TML/PDP.git
cd PDP

conda create -n pdp python=3.8
conda activate pdp
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -e .
```

Then download the Bump-em dataset from [here](https://drive.google.com/file/d/1CnlsnwA1e5U4UFqUj_Uz_l-tkkjQvh1I/view?usp=drive_link) and put it in the `data/` directory and extract it.

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


## Additional Details

### Bump-em Dataset

The dataset contains observation and action data collected on the perturbation recovery task. The observation is composed of the following quantities for 10 bodies in the model, plus a perturbation signal, yielding a 181 dimensional observation:
- 3D body position (3 * 10 values)
- 3x3 rotation matrix (9 * 10 values)
- 6D linear + angular velocities (6 * 10 values)
- Perturbation signal (1 value)

The action is the 25 DoF desired join positions. The data was collected at 50Hz.