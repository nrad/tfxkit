import logging
from tfxkit.core.logger import setup_logging
from tfxkit.core.config_loader import ConfigLoader
from tfxkit.core.data_manager import DataManager
from tfxkit.core.model_builder import ModelBuilder
from tfxkit.core.trainer import Trainer
from tfxkit.core.evaluator import Evaluator
from tfxkit.core.tuner import Tuner

logger = logging.getLogger(__name__)


class ModelFactory:
    def __init__(self):
        self.__init_components()
        self._expose(
            [
                "builder.model",
            ]
        )

    def __init_components(self, config=None, model=None, data_manager=None):
        """Initialize all components of the ModelFactory."""
        self.__init_config()
        self.__init_data_manager(config=config)
        if model is None:
            self.__init_model_builder(config=config)
            model = self.builder.model
        self.__init_trainer(config=config, model=model, data_manager=data_manager)
        self.__init_evaluator(config=config, model=model, data_manager=data_manager)
        self.__init_tuner(config=config, model=model, data_manager=data_manager)

    def __init_config(self, config_name="config", config_module="tfxkit.configs"):
        self.config_loader = ConfigLoader(
            config_name=config_name, config_module=config_module
        )
        self.config = self.config_loader.config

    def __init_data_manager(self, config):
        config = config if config else self.config
        self.data = DataManager(config)

    def __init_model_builder(self, config=None):
        config = config if config else self.config
        self.builder = ModelBuilder(self.config)

    def __init_trainer(self, config=None, model=None, data_manager=None):
        model = model if model else self.builder.model
        config = config if config else self.config
        data_manager = data_manager if data_manager else self.data
        self.trainer = Trainer(config=config, model=model, data_manager=data_manager)

    def __init_evaluator(self, config=None, model=None, data_manager=None):
        """Initialize the evaluator with the model and data manager."""
        config = config if config else self.config
        model = model if model else self.builder.model
        data_manager = data_manager if data_manager else self.data
        self.evaluator = Evaluator(config=config, model=model, data_manager=data_manager)

    def __init_tuner(self, config=None, model=None, data_manager=None):
        config = config if config else self.config
        model = model if model else self.builder.model
        data_manager = data_manager if data_manager else self.data
        self.tuner = Tuner(config=config, builder=self.builder, data=self.data)

    # def __load_data(self):
    #     self.builder = ModelBuilder(self.config)
    #     self.trainer = Trainer(self.config, self.builder.model, self.data)
    #     self.evaluator = Evaluator(self.config, self.builder.model, self.data)

    def __getattr__(self, name):
        """
        Dynamically forwards attribute access to internal components (e.g., Trainer, Evaluator).

        If exactly one component defines the requested attribute, it is returned.
        If multiple components define the same attribute, raises an AttributeError to avoid ambiguity.
        """

        required_components = ["data", "builder", "trainer", "evaluator"]
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

    def _expose(self, targets):
        """Expose selected component methods/attributes directly on the factory."""
        for dotted in targets:
            comp_name, attr_name = dotted.split(".")
            component = getattr(self, comp_name)
            setattr(self, attr_name, getattr(component, attr_name))


if __name__ == "__main__":
    logger.info("ModelFactory starting up...")
    mf = ModelFactory()
    setup_logging(level=mf.config.logging.level)
    mf.config_loader.print_config()
    # mf.model = mf.builder.model
    # define and compile the model
    # mf.model = mf.builder.define_model()
    mf.builder.compile()
    logger.info("Model Created successfully.")
    logger.info
    mf.history = mf.fit(validation_split=0.2)
    logger.info("Model training completed.")

    mf.evaluator.add_test_train_preds()
    logger.info("Predictions added to train and test datasets.")
