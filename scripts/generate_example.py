from hydra import core, initialize, compose
from omegaconf import OmegaConf
import torch
import numpy as np
from cross_view_transformer.common import setup_experiment, load_backbone
from pathlib import Path
import time
import imageio

# This file implements scripts/example.ipynb in a python file so it can be run as a script 
# (GPU allocation is tough in Jupyter Notebooks)

# You may download a pretrained model (13 Mb)
# MODEL_URL = 'https://www.cs.utexas.edu/~bzhou/cvt/cvt_nuscenes_vehicles_50k.ckpt'
# CHECKPOINT_PATH = '../logs/cvt_nuscenes_vehicles_50k.ckpt'
# !wget $MODEL_URL -O $CHECKPOINT_PATH

def main():
    # CHANGE ME
    EXPERIMENT = 'cvt_argoverse2_vehicle' # or nuscenes_vehicle
    DATASET_DIR = '/srv/datasets/argoverse2/sensor' 
    LABELS_DIR = '/srv/share2/apatni30/cvt_labels_argoverse2'
    CHECKPOINT_PATH = '/srv/share2/apatni30/raster-net/comparison/cross_view_transformers/logs/q212xtjs/checkpoints/model-v1.ckpt'
    SPLIT = 'val_qualitative_000'
    SUBSAMPLE = 5
    GIF_PATH = './predictions.gif'

    assert Path(CHECKPOINT_PATH).exists(), "Could not find path to checkpoint model"

    core.global_hydra.GlobalHydra.instance().clear()        # required for Hydra in notebooks
    initialize(config_path='../config')

    # Add additional command line overrides
    cfg = compose(
        config_name='config',
        overrides=[
            'experiment.save_dir=../logs/',                 # required for Hydra in notebooks
            f'+experiment={EXPERIMENT}',
            f'data.dataset_dir={DATASET_DIR}',
            f'data.labels_dir={LABELS_DIR}',
            'data.version=trainval',
            'loader.batch_size=1',
        ]
    )

    # resolve config references
    OmegaConf.resolve(cfg)

    print(list(cfg.keys()))
    
    model, data, viz = setup_experiment(cfg)

    dataset = data.get_split(SPLIT, loader=False)
    dataset = torch.utils.data.ConcatDataset(dataset)
    dataset = torch.utils.data.Subset(dataset, range(0, len(dataset), SUBSAMPLE))

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    print(len(dataset))

    if Path(CHECKPOINT_PATH).exists():
        network = load_backbone(CHECKPOINT_PATH)
    else:
        raise FileExistsError("Could not find path to Checkpoint Model")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network.to(device)
    network.eval()

    images = list()

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            pred = network(batch)
            visualization = np.vstack(viz(pred=pred, batch=batch,))
            images.append(visualization)

    # Save a gif
    duration = [1 for _ in images[:-1]] + [5 for _ in images[-1:]]
    imageio.mimsave(GIF_PATH, images, duration=duration)


if __name__ == "__main__":
    main()