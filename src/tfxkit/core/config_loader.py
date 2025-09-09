import logging
import sys
from hydra import initialize_config_module, compose
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)


class ConfigLoader:

    def __init__(
        self, config_name: str = "config", config_module: str = "tfxkit.configs"
    ):
        # self.config_
        self.config = self.get_config(config_name, config_module)
        # self._resolve_keys()

    # def get_config(
    #     self, config_name: str = "config", config_module: str = "tfxkit.configs"
    # ) -> DictConfig:
    #     """Load configuration using Hydra."""
    #     overrides = sys.argv[1:]  # collect CLI overrides
    #     with initialize_config_module(config_module=config_module, version_base=None):
    #         cfg = compose(config_name=config_name, overrides=overrides)
    #     # cfg = OmegaConf.to_container(cfg, resolve=True)
    #     return cfg
    def _called_from_cli(self):
        return sys.argv[0].endswith("cli.py") or sys.argv[0].endswith("cli")



    def get_config(
        self,
        config_name: str = "config",
        config_module: str = "tfxkit.configs",
        overrides: list[str] = None,
    ) -> DictConfig:
        """Load configuration using Hydra, optionally applying overrides."""
        if overrides is None:
            overrides = sys.argv[1:] if not self._called_from_cli() else []

        with initialize_config_module(config_module=config_module, version_base=None):
            return compose(config_name=config_name, overrides=overrides)

    def _resolve_keys(self, keys=None):
        # raise NotImplementedError(
        #     "This method is not implemented yet. It should resolve specific keys in the configuration."
        # )
        keys_to_resolve = [
            # "model.parameters",
            "data.train_files",
            "data.test_files",
        ]
        keys = keys if keys else keys_to_resolve
        for key in keys:
            logger.info(f"?? Resolving key: {key}")
            logger.info(f"?? Resolving key: {type(self.config)}, {self.config}")
            container = OmegaConf.to_container(
                OmegaConf.select(self.config, key), resolve=True
            )
            OmegaConf.update(self.config, key, container)
            # attr = OmegaConf.select(self.config, key)

            # if key in self.config:
            #     self.config[key] = OmegaConf.to_container(self.config[key], resolve=True)
            # else:
            #     logger.warning(f"Key '{key}' not found in config, skipping resolution.")
        # Uncomment if you want to resolve the config to a plain dictionary
        # self.config.model.parameters = OmegaConf.to_container(self.config.model.parameters, resolve=True)
        # self.config.data.train_files = OmegaConf.to_container(self.config.data.train_files, resolve=True)
        # self.config.data.test

    def print_config(self):
        """Print the configuration in YAML format."""
        # print(OmegaConf.to_yaml(self.config))
        logger.info("Configuration:")
        logger.info("\n" + OmegaConf.to_yaml(self.config))


if __name__ == "__main__":
    cl = ConfigLoader()
    cl.print_config()
