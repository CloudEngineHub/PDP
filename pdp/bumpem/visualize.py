from pdp.dataset.dataset import DiffusionPolicyDataset


def main():
    zarr_path = 'data/bumpem_dass-noise-level=0.12/data.zarr'
    dataset = DiffusionPolicyDataset(zarr_path=zarr_path, horizon=1)
    episode_iterator = dataset.get_episode_iterator()
    for episode in episode_iterator:
        import pdb; pdb.set_trace()
        print(episode)

    print('Done')



if __name__ == '__main__':
    main()