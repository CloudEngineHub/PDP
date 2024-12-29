import os
import sys
import pathlib

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
from pdp.workspace import DiffusionPolicyWorkspace


@hydra.main(
    config_path=os.path.join(str(pathlib.Path(__file__).parent), 'cfg')
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.run()
    

if __name__ == "__main__":
    main()