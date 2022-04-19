import pytorch_lightning as pl
import torch
import torch.utils.data

from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.warnings import rank_zero_warn


class VisualizationCallback(pl.Callback):
    def __init__(self, visualizers, log_image_interval=1000):
        super().__init__()

        self.visualizers = {'image': visualizers}
        self.log_image_interval = log_image_interval

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        *args,
        **kwargs
    ) -> None:
        if batch_idx % self.log_image_interval == 0:
            self._visualize_batch(outputs, trainer, batch_idx, 'train')

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        *args,
        **kwargs
    ) -> None:
        if batch_idx % self.log_image_interval == 0:
            self._visualize_batch(outputs, trainer, batch_idx, 'val')

    @rank_zero_only
    def _visualize_batch(self, outputs: STEP_OUTPUT, trainer: pl.Trainer, batch_idx: int, prefix: str):
        for key, viz in self.visualizers.items():
            self._log_image(viz(**outputs), f'{prefix}/{key}', trainer.logger)

    def _log_image(self, image_batch, tag, logger):
        if isinstance(logger, torch.utils.tensorboard.writer.SummaryWriter):
            logger.add_images(tag=tag, img_tensor=torch.from_numpy(image_batch), dataformats='NHWC')
        elif isinstance(logger, WandbLogger):
            logger.log_image(tag, image_batch)
        else:
            rank_zero_warn(f'Invalid logger {logger}')
