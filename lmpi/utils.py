import optuna
from omegaconf import DictConfig, OmegaConf


def get_config(file: str = "config.yaml", merge_cli: bool = True) -> DictConfig:
    cfg = OmegaConf.load(file)
    if merge_cli:
        cfg_cli = OmegaConf.from_cli()
        cfg = OmegaConf.merge(cfg, cfg_cli)

    return cfg


def suggest_config(trial: optuna.trial.Trial, config: DictConfig):
    suggest_dict = {}
    for k, v in config.optuna.parameters.items():
        suggest_dict[k] = getattr(trial, f"suggest_{v.type}")(name=k, **v.args)
    suggest_cfg = OmegaConf.create(suggest_dict)
    merge_cfg = OmegaConf.merge(config, suggest_cfg)
    return merge_cfg
