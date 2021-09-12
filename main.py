import os

import optuna
import pytorch_lightning as pl
from icecream import ic
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, Subset

from lmpi.data.dataset.utils import get_datasets
from lmpi.train_model import TrainModel
from lmpi.transform.utils import get_transform
from lmpi.utils import get_config, suggest_config

print = ic


def fit(config: DictConfig):
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
        **config.logger,
    )
    local_save_dir = os.path.join(
        mlflow_logger.save_dir,
        mlflow_logger.experiment_id,
        mlflow_logger.run_id,
        "artifacts",
    )
    OmegaConf.save(config, os.path.join(local_save_dir, "config.yaml"))
    checkpoint_callback = ModelCheckpoint(
        os.path.join(local_save_dir, "{epoch:02d}-{val_loss:.2f}"), monitor="val_loss"
    )
    trainer = pl.Trainer(
        **config["trainer"],
        logger=mlflow_logger,
        checkpoint_callback=checkpoint_callback,
    )
    datasets = get_datasets(
        name=config.dataset.name,
        root=config.dataset.root,
        test_size=config.dataset.test_size,
        random_state=config.dataset.random_state,
        **get_transform(),
    )

    train_dataloader = DataLoader(
        Subset(datasets["train"], range(20)),
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        Subset(datasets["val"], range(20)),
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        num_workers=4,
    )
    test_dataloader = DataLoader(
        Subset(datasets["test"], range(20)),
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        num_workers=4,
    )
    trainer.fit(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    test_loss = trainer.test(model=model, test_dataloaders=test_dataloader)[0][
        "test_loss"
    ]
    return test_loss


def objective(config: DictConfig):
    def objective_fn(trial: optuna.trial.Trial):
        config_fit = suggest_config(trial, config)
        test_loss = fit(config_fit)
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
