import logging
from tfxkit.core.logger import setup_logging
from tfxkit.core.config_loader import ConfigLoader
from tfxkit.core.data_manager import DataManager
from tfxkit.core.model_builder import ModelBuilder
from tfxkit.core.trainer import Trainer
from tfxkit.core.evaluator import Evaluator
from tfxkit.core.tuner import HyperTuner
from tfxkit.core.plotter import Plotter

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    The ModelFactory class orchestrates the full model training, and hypertuning workflow.

    It initializes core components like config, data manager, model builder, trainer,
    evaluator, and hyperparameter tuner. It also allows direct access to selected methods
    from internal components via attribute forwarding.

    Attributes:
        config (OmegaConf): Loaded Hydra config.
        builder (ModelBuilder): Constructs and compiles the model.
        data (DataManager): Provides access to training and test data.
        trainer (Trainer): Handles model training.
        evaluator (Evaluator): Evaluates model and attaches predictions.
        hyper_tuner (HyperTuner): Manages hyperparameter tuning.
    """

    def __init__(self, config=None, data_manager=None, overrides=None, debug=False):
        if not debug:
            self.__init_components(config=config, data_manager=data_manager, overrides=overrides)
            # self._expose(
            #     [
            #         "builder.model",
            #     ]
            # )

    def __init_components(self, config=None, data_manager=None, overrides=None):
        """
        Initialize all components of the ModelFactory.

        Args:
            config (OmegaConf, optional): Config object, if already loaded.
            model (keras.Model, optional): Compiled model to use instead of building from scratch.
            data_manager (DataManager, optional): Pre-initialized data manager.
        """
        self.__init_config(config=config, overrides=overrides)
        config = self.config
        if data_manager is None:
            self.__init_data_manager(config=config)
        else:
            self.data = data_manager
        # if model is None:
        self.__init_model_builder(config=config)
        builder = self.builder
        model = builder.model
        self.__init_trainer(config=config, builder=builder, data_manager=data_manager)
        self.__init_evaluator(config=config, builder=builder, data_manager=data_manager)
        self.__init_plotter(
            config=config,
            evaluator=self.evaluator,
            data_manager=data_manager,
            trainer=self.trainer,
        )
        self.__init_tuner(config=config, data_manager=data_manager)

    def __init_config(
        self,
        config=None,
        overrides=None,
    ):
        import sys, os
        from omegaconf import DictConfig

        config_kwargs = dict(
            config=None, config_dir="", config_name="", config_module="", overrides=overrides
        )

        if config is not None:
            if isinstance(config, str) and config.endswith(".yaml"):
                if not os.path.isfile(config):
                    raise FileNotFoundError(
                        f"Expected config file but did not find it: {config}"
                    )
                config_kwargs.update(
                    config_name=os.path.splitext(os.path.basename(config))[0],
                    config_dir=os.path.dirname(config),
                )
            elif isinstance(config, DictConfig):
                config_kwargs.update(config=config)
            else:
                raise NotImplementedError(
                    f"Unable to interpret the provided config: {type(config)} : {config}"
                )

        # --- check CLI overrides ---
        else:
            overrides = sys.argv[1:]
            logger.info(
                f"Looking for config_name in the command line arguments: {overrides}"
            )
            for arg in overrides:
                if arg.startswith("config_name="):
                    config_name = arg.split("=", 1)[1]
                elif arg.startswith("config_dir="):
                    config_dir = arg.split("=", 1)[1]
                elif arg.startswith("config_module="):
                    config_module = arg.split("=", 1)[1]

        self.config_loader = ConfigLoader(**config_kwargs)
        self.config = self.config_loader.config

    def __init_data_manager(self, config):
        """Initialize the data manager with the provided config."""
        config = config if config else self.config
        self.data = DataManager(config)

    def __init_model_builder(self, config=None):
        """Initialize the model builder with the provided config."""
        config = config if config else self.config
        self.builder = ModelBuilder(self.config)

    def __init_trainer(self, config=None, builder=None, data_manager=None):
        """Initialize the trainer with the builder and data manager."""
        builder = builder if builder else self.builder
        config = config if config else self.config
        data_manager = data_manager if data_manager else self.data
        self.trainer = Trainer(config=config, builder=builder, data_manager=data_manager)

    def __init_evaluator(self, config=None, builder=None, data_manager=None):
        """Initialize the evaluator with the builder and data manager."""
        config = config if config else self.config
        builder = builder if builder else self.builder
        data_manager = data_manager if data_manager else self.data
        self.evaluator = Evaluator(
            config=config, builder=builder, data_manager=data_manager
        )

    def __init_plotter(
        self, config=None, data_manager=None, evaluator=None, trainer=None
    ):
        """Initialize the plotter with the evaluator and data manager."""
        config = config if config else self.config
        evaluator = evaluator if evaluator else self.evaluator
        data_manager = data_manager if data_manager else self.data
        trainer = trainer if trainer else self.trainer
        self.plotter = Plotter(
            config=config,
            data_manager=data_manager,
            evaluator=evaluator,
            trainer=trainer,
        )

    def __init_tuner(self, config=None, builder=None, data_manager=None):
        config = config if config else self.config
        builder = builder if builder else self.builder
        data_manager = data_manager if data_manager else self.data
        self.hyper_tuner = HyperTuner(
            config=config, data=self.data
        )


    def hyper_tune(self):
        """Run the hyperparameter tuning sequence."""
        logger.info("Starting hyperparameter tuning sequence...")
        logger.info("Hyper Tunning Sequence {self.hyper_tuner.config.tuner.sequence}")
        self.hyper_tuner.run_sequence()

    def attach_predictions(self):
        """Attach model predictions to the training and test datasets."""
        self.evaluator.add_test_train_preds()

    def make_plots(self):
        """Generate and save all configured plots."""
        self.attach_predictions()
        self.plotter.run_sequence()



    def fit(self, save_path=None, overwrite=None, verbose=0):
        """
        Runs the model training and ensures automatic saving of the model and weights
        after the training.

        Args:
            save_path (str or Path, optional): Where to save the trained model.
                If None, the path from `config.info.save_dir` will be used.

        Returns:
            History: The Keras History object from `model.fit()`.

        TODO: pass on **kwargs to trainer.fit and save the updated config file
        """
        history = self.trainer.fit(verbose=verbose)
        self.builder.save_model(save_path, overwrite=overwrite)
        return history

    def __getattr__(self, name):
        """
        Dynamically forwards attribute access to internal components (e.g., Trainer, Evaluator).

        If exactly one component defines the requested attribute, it is returned.
        If multiple components define the same attribute, raises an AttributeError to avoid ambiguity.
        """

        required_components = ["data", "builder", "trainer", "evaluator", "hyper_tuner"]
        components = []
        for comp_name in required_components:
            # component = getattr(self, comp_name, None)
            if comp_name not in dir(self):
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{comp_name}'\n"
                    "Most likely, the component was not initialized."
                )
            component = getattr(self, comp_name)
            components.append(component)

        matches = [getattr(c, name) for c in components if hasattr(c, name)]

        if len(matches) > 1:
            raise AttributeError(
                f"Ambiguous attribute '{name}' found in multiple components: {', '.join(c.__class__.__name__ for c in components)}"
            )

        elif matches:
            logger.debug(f"Accessing attribute '{name}' from {matches[0].__class__}")
            return matches[0]

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    # def _expose(self, targets):
    #     """Expose selected component methods/attributes directly on the factory."""
    #     for dotted in targets:
    #         comp_name, attr_name = dotted.split(".")
    #         component = getattr(self, comp_name)
    #         setattr(self, attr_name, getattr(component, attr_name))

    @property
    def model(self):
        return self.builder.model

    def clone_factory(self, *, config=None, overrides=None):
        """
        Create a new ModelFactory instance with an updated configuration.
        Args:
            config (OmegaConf, optional): New configuration to use. If None, uses the current config.
            overrides (list of str, optional): List of dotlist strings to override specific config, eg.
                ['key1=value1', 'item1.key2=value2'].
        Returns:
            ModelFactory: A new instance of ModelFactory with the updated configuration.
        """
        new_config = config if config is not None else self.config
        if overrides:
            from omegaconf import OmegaConf
            override_conf = OmegaConf.from_dotlist(overrides)
            new_config = OmegaConf.merge(new_config, override_conf)
        return ModelFactory(config=new_config)

    # def NewFactory(self, config=None, overrides=None):
    #     new_config =


# @hydra.main(config_path="../../examples/configs", config_name="quickstart", version_base=None)
# def main(cfg: DictConfig):
#     setup_logging(level=cfg.logging.level)

#     mf = ModelFactory(config=cfg)
#     mf.builder.compile()
#     mf.history = mf.trainer.fit(
#         validation_split=cfg.training.validation_split,
#         batch_size=cfg.training.batch_size,
#         epochs=cfg.training.epochs,
#         sample_weight=mf.data.sample_weight_train,
#     )
#     mf.evaluator.add_test_train_preds()


if __name__ == "__main__":
    logger.info("ModelFactory starting up...")

    import argparse

    parser = argparse.ArgumentParser(
        description="Run ModelFactory with optional config path."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to a custom config file (YAML).",
    )
    args = parser.parse_args()

    logger.info("ModelFactory starting up...")

    mf = ModelFactory(config=args.config_path)
    setup_logging(level=mf.config.logging.level)
    mf.config_loader.print_config()
    mf.builder.compile()
    logger.info("Model Created successfully.")
    mf.history = mf.trainer.fit(
        validation_split=mf.config.training.validation_split,
        batch_size=mf.config.training.batch_size,
        epochs=mf.config.training.epochs,
        sample_weight=mf.data.sample_weight_train,
    )

    logger.info("Model training completed.")

    mf.evaluator.add_test_train_preds()
    logger.info("Predictions added to train and test datasets.")
