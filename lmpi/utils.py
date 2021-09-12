import optuna
from omegaconf import DictConfig, OmegaConf


def get_config(file: str = "config.yaml", merge_cli: bool = True) -> DictConfig:
    cfg = OmegaConf.load(file)
    if merge_cli:
        cfg_cli = OmegaConf.from_cli()
        cfg = OmegaConf.merge(cfg, cfg_cli)

    return cfg


def make_suggest_config(trial: optuna.trial.Trial, config: DictConfig) -> DictConfig:
    suggest_config = OmegaConf.create({})
    OmegaConf.set_struct(suggest_config, True)

    def dfs(trial, cur: DictConfig, suggest_config: DictConfig):
        for k, v in cur.items():
            if v.get("type", False):
                suggest_func = getattr(trial, f"suggest_{v.type}")
                suggest_arg = suggest_func(name=k, **v.args)
                OmegaConf.update(suggest_config, k, suggest_arg, force_add=True)
            else:
                OmegaConf.update(suggest_config, k, {}, force_add=True)
                dfs(trial, cur[k], suggest_config[k])

    dfs(trial, config, suggest_config)
    return suggest_config


def update_suggest_config(trial: optuna.trial.Trial, config: DictConfig) -> DictConfig:
    suggest_config = make_suggest_config(trial, config.optuna.parameters)
    OmegaConf.set_struct(config, False)
    merge_config = OmegaConf.unsafe_merge(config, suggest_config)
    return merge_config
