import os
import pathlib

import numba
import numpy as np
import torch
from torch.utils.data import Dataset

from pdp.dataset.replay_buffer import ReplayBuffer
from pdp.utils.data import dict_apply
from pdp.utils.normalizer import LinearNormalizer

# Get the top-level directory of the project
PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]


@numba.jit(nopython=True)
def create_indices(episode_ends, sequence_length, pad_before=0, pad_after=0):
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0: start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx
            ])

    indices = np.array(indices)
    return indices


class DiffusionPolicyDataset(Dataset):
    def __init__(self, zarr_path, horizon=1, pad_before=0, pad_after=0):
        self.keys = ['obs', 'action']
        self.meta_keys = ['motion_fname']
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        zarr_path = os.path.join(PROJECT_DIR, zarr_path)
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=self.keys)
        self.episode_ends = self.replay_buffer.episode_ends[:]
        self.episode_indices = self.replay_buffer.get_episode_idxs()
        self.indices = create_indices(
            self.episode_ends,
            sequence_length=self.horizon, 
            pad_before=pad_before, 
            pad_after=pad_after
        )
    
    def get_normalizer(self):
        data = {
            'obs': self.replay_buffer['obs'],
            'action': self.replay_buffer['action'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, mode='limits')
        return normalizer

    @property
    def num_episodes(self):
        return self.replay_buffer.num_episodes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.sample_sequence(idx)
        data = {    
            'obs': sample['obs'],           # T, D_o
            'action': sample['action'],     # T, D_a
            'motion_fname': sample['motion_fname'],
        }
        data = dict_apply(data, torch.from_numpy)
        data = dict_apply(data, lambda x: x.to(torch.float32))
        return data

    def get_validation_dataset(self):
        raise NotImplementedError

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx  = self.indices[idx]
        result = dict()
        for key in self.keys:
            buff = self.replay_buffer[key]
            sample = buff[buffer_start_idx:buffer_end_idx]
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.horizon):
                data = np.zeros((self.horizon, *buff.shape[1:]), dtype=buff.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.horizon:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample

            result[key] = data

        ep_idx = self.episode_indices[buffer_start_idx]
        for key in self.meta_keys:
            metadata_buff = self.replay_buffer.meta[key]
            result[key] = metadata_buff[ep_idx]
        
        return result

    def get_episode_iterator(self):
        for ep_idx in range(self.num_episodes):
            start = 0
            if ep_idx > 0: start = self.episode_ends[ep_idx-1]
            end = self.episode_ends[ep_idx]
            ep_data = dict()
            for key in self.keys:
                ep_data[key] = self.replay_buffer[key][start:end]

            for key in self.meta_keys:
                ep_data[key] = self.replay_buffer.meta[key][ep_idx]
            
            yield ep_data
