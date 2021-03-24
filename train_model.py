import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision
from icecream import ic
from sklearn import preprocessing
import numpy as np
import pandas as pd
import glob
import os, sys, re
import omegaconf
sys.path.append(os.path.abspath("."))
from models import encoder, decoder


class TrainModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder()
        self.decoder = decoder(cfg.num_class)
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x):
        h = self.encoder(x)
        res = self.decoder(h)
        return h, res

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), **self.cfg.optimizer)
        if self.cfg.on_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, 
                            **self.cfg.scheduler)
            return [optimizer], [scheduler]
        else:
            return optimizer
    
    # fit
    def training_step(self, batch, batch_nb):
        x, y = batch
        _, y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    # val
    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        _, y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)

