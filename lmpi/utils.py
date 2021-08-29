from omegaconf import OmegaConf, DictConfig


def get_config(file: str = "config.yaml",
               merge_cli: bool = True) -> DictConfig:
    cfg = OmegaConf.load(file)
    if merge_cli:
        cfg_cli = OmegaConf.from_cli()
        cfg = OmegaConf.merge(cfg, cfg_cli)

    return cfg
