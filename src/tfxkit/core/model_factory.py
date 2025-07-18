import logging
from tfxkit.core.logger import setup_logging
from tfxkit.core.config_loader import ConfigLoader
from tfxkit.core.data_manager import DataManager
from tfxkit.core.model_builder import ModelBuilder
from tfxkit.core.trainer import Trainer
from tfxkit.core.evaluator import Evaluator

logger = logging.getLogger(__name__)


class ModelFactory:
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.config
        self.data = DataManager(self.config)
        self.builder = ModelBuilder(self.config)
        # self.buil = self.model_builder

        self.trainer = Trainer(self.config, self.builder.model, self.data)
        self.evaluator = Evaluator(self.config, self.builder.model, self.data)

    def __getattr__(self, name):
        """
        Dynamically forwards attribute access to internal components (e.g., Trainer, Evaluator).

        If exactly one component defines the requested attribute, it is returned.
        If multiple components define the same attribute, raises an AttributeError to avoid ambiguity.
        """
        components = [self.trainer, self.evaluator]
        matches = [getattr(c, name) for c in components if hasattr(c, name)]

        if len(matches) > 1:
            raise AttributeError(
                f"Ambiguous attribute '{name}' found in multiple components: {', '.join(c.__class__.__name__ for c in components)}"
            )

        elif matches:
            logger.debug(
                f"Accessing attribute '{name}' from {matches[0].__class__.__name__}"
            )
            return matches[0]

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
    @forward("trainer")
    def model(*args, **kwargs):

        """Delegates to the Trainer's model method."""
        pass


if __name__ == "__main__":
    logger.info("ModelFactory starting up...")
    mf = ModelFactory()
    setup_logging(level=mf.config.logging.level)
    mf.config_loader.print_config()

    # define and compile the model
    # mf.model = mf.builder.define_model()
    mf.builder.compile()
    logger.info("Model Created successfully.")

    mf.fit(epochs=2)
    logger.info("Model training completed.")

    mf.evaluator.add_test_train_preds()
    logger.info("Predictions added to train and test datasets.")

    #
