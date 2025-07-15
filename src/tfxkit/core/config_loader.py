import logging
import sys
from hydra import initialize_config_module, compose
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)

class ConfigLoader:

    def __init__(self, config_name: str = "config", config_module: str = "tfxkit.configs"):
        self.config = self.get_config(config_name, config_module)

    def get_config(self, config_name: str = "config", config_module: str = "tfxkit.configs") -> DictConfig:
        """Load configuration using Hydra."""
        overrides = sys.argv[1:]  # collect CLI overrides
        with initialize_config_module(config_module=config_module, version_base=None):
            cfg = compose(config_name=config_name, overrides=overrides)
        cfg = OmegaConf.to_container(cfg, resolve=True)
        return cfg

    def print_config(self):
        """Print the configuration in YAML format."""
        # print(OmegaConf.to_yaml(self.config))
        logger.info("Configuration:")
        logger.info("\n"+OmegaConf.to_yaml(self.config))


if __name__ == "__main__":
    cl = ConfigLoader()
    cl.print_config()
