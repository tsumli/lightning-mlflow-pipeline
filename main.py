import os

import optuna
import pytorch_lightning as pl
from icecream import ic
from lmpi.data.dataset import MNIST_datasets
from lmpi.train_model import TrainModel
from lmpi.utils import get_config
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

print = ic


def fit(cfg: DictConfig):
    """
    Parameters
    ----------
    cfg:
    config for fitting

    Returns
    ----------
    test_loss: Any

    """
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
    test_loss = trainer.test(net, dl_test)[0]["test_loss"]
    return test_loss


def objective(config: DictConfig):
    def suggest_config(trial):
        suggest_dict = {}
        for k, v in config.optuna.parameters.items():
            suggest_dict[k] = getattr(trial, f"suggest_{v.type}")(name=k, **v.args)
        suggest_cfg = OmegaConf.create(suggest_dict)
        merge_cfg = OmegaConf.merge(config, suggest_cfg)
        return merge_cfg

    def objective_fn(trial):
        config = suggest_config(trial)
        test_loss = fit(config)
        return test_loss

    return objective_fn


def main():
    cfg = get_config(file="config.yaml", merge_cli=True)

    if hasattr(cfg, "seed"):
        seed_everything(cfg.seed)

    if cfg.optuna.enable:
        study = optuna.create_study(
            direction=cfg.optuna.create_study.direction,
            storage=cfg.optuna.create_study.storage,
            study_name=cfg.optuna.create_study.study_name,
            load_if_exists=True,
        )

        study.optimize(
            func=objective(cfg),
            n_trials=cfg.optuna.n_trials,
        )
        best_params = study.best_params
        print(best_params)

    else:
        test_loss = fit(cfg)
        print(test_loss)


if __name__ == "__main__":
    main()
