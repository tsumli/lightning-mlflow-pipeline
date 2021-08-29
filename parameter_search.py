import optuna
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from main import fit


def get_config(file: str = "config.yaml"):
    cfg = OmegaConf.load(file)
    cfg_cli = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cfg_cli)
    seed_everything(cfg.seed)
    return cfg


def get_config_optuna(file: str = "config_optuna.yaml"):
    cfg = OmegaConf.load(file)
    return cfg


def mod_params(trial, cfg):
    lr = trial.suggest_float("lr", *cfg.optuna.lr)
    return OmegaConf.create({"optimizer": {"lr": lr}})


def objective(trial, cfg, cfg_optuna):
    cfg_optuna = mod_params(trial, cfg_optuna)
    cfg = OmegaConf.merge(cfg, cfg_optuna)
    test_loss = fit(cfg)
    return test_loss


def main():
    cfg = get_config()
    cfg_optuna = get_config_optuna()
    study = optuna.create_study(
        direction="maximize",
        storage=cfg_optuna.optuna_study.storage,
        study_name="optuna",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, cfg, cfg_optuna),
        n_trials=cfg_optuna.optuna.n_trials,
    )
    best_params = study.best_params
    print(f"best_params: {best_params}")


if __name__ == "__main__":
    main()
