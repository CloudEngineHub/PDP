import argparse
import collections
from pathlib import Path

import imageio
import dill
import hydra
import torch
import numpy as np
from pdp.bumpem.env import Skeleton


def load_checkpoint(payload):
    cfg = payload['cfg']
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.load_payload(payload)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval()
    return policy


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.env = Skeleton()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        payload = torch.load(open(args.ckpt_path, 'rb'), pickle_module=dill)
        self.cfg = payload['cfg']
        self.policy = load_checkpoint(payload)
        self.policy = self.policy.to(self.device)

    def run(self):
        obs = self.env.reset()
        T_obs = self.policy.T_obs
        obs_hist = collections.deque([obs.copy() for _ in range(T_obs)], maxlen=T_obs)

        frames = []
        n_steps = 1
        with torch.no_grad():
            while True:
                if n_steps % 10 == 0:
                    print(f'Step {n_steps}, Env time: {self.env.data.time:.2f}')

                action_dict = self.policy.predict_action({
                    'obs': torch.tensor(
                        np.vstack(obs_hist)
                    ).unsqueeze(0).to(self.device, dtype=torch.float32)
                })
                action = action_dict['action'][0, 0, :]
                obs, reward, done, info = self.env.step(action.cpu().numpy())
                if done:
                    break

                obs_hist.append(obs)
                n_steps += 1
                if self.args.save_video:
                    frames.append(info['rgb'])

        print(f'Episode finished after {n_steps} steps')

        if self.args.save_video:
            Path('visuals').mkdir(exist_ok=True)
            imageio.mimsave('visuals/bumpem_eval_result.mp4', frames, fps=50)



def main(args):
    evaluator = Evaluator(args)
    evaluator.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--save_video', action='store_true')
    args = parser.parse_args()

    main(args)