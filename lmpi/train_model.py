import pytorch_lightning as pl
import torch
from torch.nn import CrossEntropyLoss

from .models.model import decoder, encoder


class TrainModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder()
        self.decoder = decoder(cfg.num_class)
        self.loss = CrossEntropyLoss()

    def forward(self, x):
        h = self.encoder(x)
        res = self.decoder(h)
        return h, res

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), **self.cfg.optimizer)
        if self.cfg.optimizer.scheduler.enable:
            cfg_scheduler = self.cfg.optimizer.scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **cfg_scheduler
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_nb):
        x, y = batch
        _, y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        _, y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss)
