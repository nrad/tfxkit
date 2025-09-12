import keras_tuner as kt
import numpy as np
import logging
from omegaconf import OmegaConf
from tfxkit.common.base_utils import import_function
from tfxkit.core.model_builder import ModelBuilder
from pydoc import locate

logger = logging.getLogger(__name__)


class HyperTuner:
    """
    Controller class for executing a sequence of hyperparameter tuning steps using Keras Tuner.

    This class iterates over a sequence of tuning steps defined in the configuration
    and applies each one by dynamically importing and calling the specified function.
    Each function builds a model and runs a tuner with custom settings.

    Attributes:
        config (OmegaConf): The full configuration object.
        builder (ModelBuilder): Responsible for compiling models.
        data (Namespace): Holds X_train, y_train, and optionally sample_weight_train.

    Methods:
        run_sequence():
            Executes all tuner functions defined in the config's sequence.
    """
    def __init__(self, config, builder, data):
        self.config = config
        self.builder = builder
        self.data = data
        self.results = []
        self.tuner_config = self.config.get("tuner")

    def locate_function(self, func_path):
        if hasattr(self, func_path):
            return getattr(self, func_path)
        else:
            func = locate(func_path)
            if func is None:
                raise ImportError(f"Could not locate function '{func_path}'")
            return func


    def run_sequence(self):
        self.tuners = []
        for name in self.config.tuner.sequence:
            func_config = self.config.tuner.functions[name]
            func_path = func_config.function
            logging.debug(f"Importing function: {func_path}")
            print(f"Importing function: {func_path}")
            func = locate(func_path)
            if func is None:
                raise ImportError(f"Could not locate function '{func_path}'")

            kwargs = OmegaConf.to_container(func_config.parameters, resolve=True)
            settings = (
                OmegaConf.to_container(func_config.get("settings"), resolve=True)
                if func_config.get("settings")
                else {}
            )

            logger.info(f"Running tuner step: {name}")
            # func(self, **kwargs, **settings)
            model_builder = func(
                config=self.config,
                builder=self.builder,
                data=self.data,
                **kwargs,
                **settings,
            )
            # self.results.append(result)

            tuner_func = self.locate_function(self.tuner_config.tuner.function)
            tuner_params = OmegaConf.to_container(self.tuner_config.tuner.parameters, resolve=True)
            logger.info(f"Using tuner function: {tuner_func.__name__} with params: {tuner_params}")
            tuner = tuner_func(
                hypermodel=model_builder, **tuner_params
            )
            tuner = tuner_func(model_builder, **tuner_params)

            search_params = OmegaConf.to_container(self.tuner_config.search, resolve=True)
            validation_split = search_params.get("validation_split")
            if not validation_split:
                validation_split = self.config.training.get("validation_split", None)
            search_params.update(validation_split=validation_split)
            tuner.search(
                self.data.X_train,
                self.data.y_train,
                sample_weight=self.data.sample_weight_train,
                **search_params,
                # batch_size=50_000,
                # verbose=0,
            )
            self.tuners.append((name, tuner))

        logger.info("Tuning sequence completed.")

def generic_tuner(
    config,
    builder,
    data,
    **kwargs,
):
    """A generic tuner that can tune any hyperparameter specified in kwargs.
    kwargs should be a dictionary where keys are the hyperparameter names,
    corresponding to config fields and values are lists of possible values.

    The modified config file is attached to the compiled model as model.config
    so that it can be saved and inspected later.

    """

    def build_model(hp):
        new_conf = config.copy()
        for kwarg, choices in kwargs.items():
            value = hp.Choice(kwarg, choices)
            OmegaConf.update(new_conf, kwarg, value)
            logger.debug(f"Setting {kwarg} to {value}")
        logger.info(f"Building model with config: {new_conf}")
        builder = ModelBuilder(new_conf)

        model = builder.compile()
        model.config = new_conf #
        return model

    return build_model




def optimizer_tuning(
    config,
    builder,
    data,
    optimizers=["keras.optimizers.adam", "keras.optimizers.sgd", "rmsprop"],
    # learning_rates=[1e-2, 1e-3, 1e-4],
):
    """Tune the optimizer and learning rate."""

    def build_model(hp):
        optimizer = hp.Choice("optimizer", optimizers)
        new_conf = config.copy()
        OmegaConf.update(new_conf, "optimizer.function", "keras.optimizers.Adam")
        builder = ModelBuilder(new_conf)
        return builder.compile()

    return build_model


