from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
import hydra
import tqdm
import cv2

from omegaconf import DictConfig
from pytorch_lightning.core.memory import ModelSummary
from cross_view_transformer.common import setup_config, setup_experiment


def setup(cfg):
    cfg.data.augment = 'none'
    cfg.loader.num_workers = 0


def run_fake(data, model, optim, viz_func):
    device = torch.device('cuda')

    model = model.to(device)
    model.train()

    sample = next(iter(data))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}

    for i in tqdm.tqdm(range(100+1)):
        output = model.shared_step(batch)

        loss = output['loss']
        loss.backward()

        # alert if no grads
        if i == 0:
            alert = [k for k, v in model.named_parameters() if v.grad is None]

            if alert:
                print(alert)

        optim.step()
        optim.zero_grad()

        if i % 10 == 0:
            print(model.metrics.compute())

            model.metrics.reset()

            with torch.no_grad():
                yield np.vstack(viz_func(**output))


@hydra.main(config_path=Path.cwd() / 'config', config_name='config.yaml')
def main(cfg: DictConfig):
    setup_config(cfg, setup)

    pl.seed_everything(2022, workers=True)

    model, data, viz_fn = setup_experiment(cfg)
    optim = model.configure_optimizers(disable_scheduler=True)[0][0]
    loader = data.train_dataloader(shuffle=True)

    print(ModelSummary(model, mode='full'))

    with torch.autograd.detect_anomaly():
        with torch.cuda.amp.autocast(enabled=False):
            for viz in run_fake(loader, model, optim, viz_fn):
                cv2.imshow('debug', cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)


if __name__ == '__main__':
    main()
