import hydra
from omegaconf import DictConfig, OmegaConf

import brainwhisperer.config.schema  # noqa: F401 — registers BWConfig in ConfigStore


@hydra.main(config_path="brainwhisperer/config", config_name="base", version_base="1.3")
def main(cfg: DictConfig) -> None:
    OmegaConf.to_container(cfg, throw_on_missing=True)
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
