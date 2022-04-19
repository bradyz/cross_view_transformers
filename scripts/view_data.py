import hydra
import numpy as np
import cv2

from pathlib import Path
from tqdm import tqdm

from cross_view_transformer.common import setup_config, setup_data_module, setup_viz


def setup(cfg):
    print('See training set by adding +split=train')
    print('Shuffle samples by adding +shuffle=false')

    cfg.loader.batch_size = 1

    if 'split' not in cfg:
        cfg.split = 'val'

    if 'shuffle' not in cfg:
        cfg.shuffle = False


@hydra.main(config_path=Path.cwd() / 'config', config_name='config.yaml')
def main(cfg):
    setup_config(cfg, setup)

    data = setup_data_module(cfg)
    viz = setup_viz(cfg)
    loader = data.get_split(cfg.split, shuffle=cfg.shuffle)

    print(f'{cfg.split}: {len(loader)} total samples')

    for batch in tqdm(loader):
        img = np.vstack(viz(batch))

        cv2.imshow('debug', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
