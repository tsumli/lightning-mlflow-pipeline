import os
from typing import Optional

import optuna
import pytorch_lightning as pl
from icecream import ic
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from lmpi.train_model import TrainModel
from lmpi.data.dataloader import make_dataloader
from lmpi.utils import get_config, update_suggest_config

print = ic


def fit(config: DictConfig, trial: Optional[optuna.trial.Trial]):
    """
    Parameters
    ----------
    config:
    config for fitting

    Returns
    ----------
    test_loss: Any

    """
    model = TrainModel(config)
    mlflow_logger = MLFlowLogger(
        tags={"trial": trial.number} if trial is not None else None,
        **config.logger,
    )
    local_save_dir = os.path.join(
        mlflow_logger.save_dir,
        mlflow_logger.experiment_id,
        mlflow_logger.run_id,
        "artifacts",
    )
    OmegaConf.save(
        config,
        os.path.join(local_save_dir, "config.yaml")
    )
    checkpoint_callback = ModelCheckpoint(
        os.path.join(local_save_dir, "{epoch:02d}-{val_loss:.2f}"),
        monitor="val_loss"
    )
    trainer = pl.Trainer(
        **config["trainer"],
        logger=mlflow_logger,
        callbacks=[checkpoint_callback],
    )

    dataloader = make_dataloader(config)

    trainer.fit(
        model=model,
        train_dataloader=dataloader["train"],
        val_dataloaders=dataloader["val"],
    )
    test_loss = trainer.test(
        model=model,
        test_dataloaders=dataloader["test"]
    )
    test_loss = test_loss[0]["test_loss"]
    return test_loss


def objective(config: DictConfig):
    def objective_fn(trial: optuna.trial.Trial):
        config_fit = update_suggest_config(trial, config)
        test_loss = fit(config_fit, trial)
        return test_loss

    return objective_fn


def main():
    config = get_config(file="config.yaml", merge_cli=True)

    if hasattr(config, "seed"):
        seed_everything(config.seed)

    if config.optuna.enable:
        study = optuna.create_study(
            direction=config.optuna.create_study.direction,
            storage=config.optuna.create_study.storage,
            study_name=config.optuna.create_study.study_name,
            load_if_exists=True,
        )

        study.optimize(
            func=objective(config),
            n_trials=config.optuna.n_trials,
        )
        best_params = study.best_params
        print(best_params)

    else:
        test_loss = fit(config)
        print(test_loss)


if __name__ == "__main__":
    main()
