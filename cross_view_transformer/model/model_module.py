import torch
import pytorch_lightning as pl


class ModelModule(pl.LightningModule):
    def __init__(self, backbone, loss_func, metrics, optimizer_args, scheduler_args=None, cfg=None):
        super().__init__()

        self.save_hyperparameters(
            cfg,
            ignore=['backbone', 'loss_func', 'metrics', 'optimizer_args', 'scheduler_args'])

        self.backbone = backbone
        self.loss_func = loss_func
        self.metrics = metrics

        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args

    def forward(self, batch):
        return self.backbone(batch)

    def shared_step(self, batch, prefix='', on_step=False, return_output=True):
        pred = self(batch)
        loss, loss_details = self.loss_func(pred, batch)

        self.metrics.update(pred, batch)

        if self.trainer is not None:
            self.log(f'{prefix}/loss', loss.detach(), on_step=on_step, on_epoch=True)
            self.log_dict({f'{prefix}/loss/{k}': v.detach() for k, v in loss_details.items()},
                          on_step=on_step, on_epoch=True)

        # Used for visualizations
        if return_output:
            return {'loss': loss, 'batch': batch, 'pred': pred}

        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train', True,
                                batch_idx % self.hparams.experiment.log_image_interval == 0)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val', False,
                                batch_idx % self.hparams.experiment.log_image_interval == 0)

    def on_validation_start(self) -> None:
        self._log_epoch_metrics('train')
        self._enable_dataloader_shuffle(self.trainer.val_dataloaders)

    def validation_epoch_end(self, outputs):
        self._log_epoch_metrics('val')

    def _log_epoch_metrics(self, prefix: str):
        """
        lightning is a little odd - it goes

        on_train_start
        ... does all the training steps ...
        on_validation_start
        ... does all the validation steps ...
        on_validation_epoch_end
        on_train_epoch_end
        """
        metrics = self.metrics.compute()

        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, val in value.items():
                    self.log(f'{prefix}/metrics/{key}{subkey}', val)
            else:
                self.log(f'{prefix}/metrics/{key}', value)

        self.metrics.reset()

    def _enable_dataloader_shuffle(self, dataloaders):
        """
        HACK for https://github.com/PyTorchLightning/pytorch-lightning/issues/11054
        """
        for v in dataloaders:
            v.sampler.shuffle = True
            v.sampler.set_epoch(self.current_epoch)

    def configure_optimizers(self, disable_scheduler=False):
        parameters = [x for x in self.backbone.parameters() if x.requires_grad]
        optimizer = torch.optim.AdamW(parameters, **self.optimizer_args)

        if disable_scheduler or self.scheduler_args is None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda lr: 1)
        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **self.scheduler_args)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
