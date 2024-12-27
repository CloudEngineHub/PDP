from pdp.dataset.dataset import DiffusionPolicyDataset
from pdp.bumpem.env import Skeleton


def main():
    # zarr_path = 'data/bumpem_dass-noise-level=0.12/data.zarr'
    # dataset = DiffusionPolicyDataset(zarr_path=zarr_path, horizon=1)
    # episode_iterator = dataset.get_episode_iterator()
    
    env = Skeleton()
    import pdb; pdb.set_trace()


    print('Done')


if __name__ == '__main__':
    main()