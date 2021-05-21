import os

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from dataset.dataset import MNIST_datasets
from train_model import TrainModel


def get_config(file: str = "config.yaml"):
    cfg = OmegaConf.load(file)
    cfg_cli = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cfg_cli)
    seed_everything(cfg.seed)
    return cfg


def fit(cfg) -> None:
    net = TrainModel(cfg)
    mlflow_logger = MLFlowLogger(
        **cfg.logger,
    )
    mlflow_logger.log_hyperparams(cfg)
    local_save_dir = os.path.join(
        mlflow_logger.save_dir,
        mlflow_logger.experiment_id,
        mlflow_logger.run_id,
        "artifacts",
    )
    OmegaConf.save(cfg, os.path.join(local_save_dir, "config.yaml"))
    checkpoint_callback = ModelCheckpoint(
        os.path.join(local_save_dir, "{epoch:02d}-{val_loss:.2f}"), monitor="val_loss"
    )
    trainer = pl.Trainer(
        **cfg["trainer"], logger=mlflow_logger, checkpoint_callback=checkpoint_callback
    )
    dl_train, dl_val, dl_test = MNIST_datasets()
    trainer.fit(net, dl_train, dl_val)
    trainer.test(net, dl_test)


if __name__ == "__main__":
    cfg = get_config()
    fit(cfg)
