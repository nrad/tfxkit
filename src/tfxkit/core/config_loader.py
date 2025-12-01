import logging
import sys
import os
from hydra import initialize_config_module, compose, initialize_config_dir
from omegaconf import OmegaConf, DictConfig
from typing import Optional

logger = logging.getLogger(__name__)


class ConfigLoader:

    def __init__(
        self,
        config: Optional[DictConfig] = None,
        config_name: str = "",
        config_module: str = "",
        config_dir: str = "",
        overrides: list[str] = None,
    ):
        """
        Intialize the ConfigLoader.
        If config is provided, it is used directly (optionally merged with overrides).
        overrides should be a list of strings like ['key1=value1', 'item1.key2=value2'].
        """
        if config is not None:
            print("config:", config)
            if overrides:
                # OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
                config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
            self.config = config

            if config_name or config_dir or config_module:
                raise ValueError(
                    "config was already provided. Config name should not be also given."
                )

        else:
            self.config = self.get_config(
                config_name,
                config_module,
                config_dir,
                overrides=overrides,
            )

    @property
    def save_dir(self):
        return self.config.info.save_dir

    @property
    def model_name(self):
        return self.config.info.model_name

    @property
    def model_dir(self):
        return os.path.join(self.save_dir, self.model_name)

    @property
    def config_path(self):
        return os.path.join(self.model_dir, "config.yaml")



    def _called_from_cli(self):
        return sys.argv[0].endswith("cli.py") or sys.argv[0].endswith("cli")

    # @staticmethod
    def get_config(
        self,
        config_name: str = "config",
        config_module: str = "",
        config_dir: str = "",
        overrides: list[str] = None,
    ) -> DictConfig:
        """Load configuration using Hydra, optionally applying overrides."""

        if not config_module and not config_dir:
            raise ValueError(
                f"Either config_module or config_dir must be provided. Instead got: \n{config_dir = }\n{config_module = }"
            )
        if config_module and config_dir:
            raise ValueError(
                "Only one of config_module or config_dir should be provided."
            )
        if config_dir:
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                return compose(config_name=config_name, overrides=overrides)
        elif config_module:
            with initialize_config_module(
                config_module=config_module, version_base=None
            ):
                return compose(config_name=config_name, overrides=overrides)

    def _resolve_keys(self, keys=None):
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

    def print_config(self, config=None):
        """Print the configuration in YAML format."""
        # print(OmegaConf.to_yaml(self.config))
        config = config if config else self.config
        logger.info("Configuration:")
        logger.info("\n" + OmegaConf.to_yaml(self.config))

    def save_config(self, file_path: str = None, config=None):
        """Save the configuration to a YAML file."""
        config = config if config else self.config
        file_path = file_path if file_path else self.config_path
        if not os.path.isdir(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            OmegaConf.save(config, file)
        logger.info(f"Configuration saved to {file_path}")




if __name__ == "__main__":
    cl = ConfigLoader()
    cl.print_config()
