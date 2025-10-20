import keras_tuner as kt
import numpy as np
import logging
from omegaconf import OmegaConf
from tfxkit.common.base_utils import import_function
from tfxkit.core.model_builder import ModelBuilder
from pydoc import locate

import os

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
    def __init__(self, config, data):
        self._config = config
        # self._builder = builder
        self._data = data
        self.results = []
        self.tuner_config = self._config.get("tuner")

    def locate_function(self, func_path):
        if hasattr(self, func_path):
            return getattr(self, func_path)
        else:
            func = locate(func_path)
            if func is None:
                raise ImportError(f"Could not locate function '{func_path}'")
            return func

    @property
    def sequence(self):
        return self.tuner_config.get("sequence", [])

    @property
    def directory(self):
        dir_path = self._config.tuner.tuner.get("parameters", {}).get("directory", None)
        if dir_path is None:
            save_dir = self._config.info.get("save_dir", None)
            model_name = self._config.info.get("model_name", None)
            if model_name is None:
                raise ValueError("No model name specified. Please set 'info.model_name' in the config.")
            dir_path = os.path.join(save_dir, model_name, "HPTunning")
        if dir_path is None:
            raise ValueError("No directory specified for tuner results. Please either set \
                             'tuner.tuner.parameters.directory' \or 'save_dir' in the config.")
        return dir_path

    def get_best_models(self, num_models=1):
        if not hasattr(self, "tuners"):
            raise ValueError("No tuners found. Please run 'run_sequence()' first.")
        best_models = []
        for name, tuner in self.tuners:
            models = tuner.get_best_models(num_models=num_models)
            best_models.append((name, models))
        return best_models

    def run_sequence(self):
        self.tuners = []
        for name in self.sequence:
            func_config = self._config.tuner.functions[name]
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
                config=self._config,
                # builder=self._builder,
                data=self._data,
                **kwargs,
                **settings,
            )
            # self.results.append(result)

            tuner_func = self.locate_function(self.tuner_config.tuner.function)
            tuner_params = OmegaConf.to_container(self.tuner_config.tuner.parameters, resolve=True)
            tuner_params["directory"] = self.directory
            logger.info(f"Using tuner function: {tuner_func.__name__} with params: {tuner_params}")

            tuner = tuner_func(
                hypermodel=model_builder, **tuner_params
            )

            search_params = OmegaConf.to_container(self.tuner_config.search, resolve=True)
            validation_split = search_params.get("validation_split")
            if validation_split is None:
                validation_split = self._config.training.get("validation_split", None)
            search_params.update(validation_split=validation_split)            
            tuner.search(
                self._data.X_train,
                self._data.y_train,
                sample_weight=self._data.sample_weight_train,
                **search_params,
            )
            self.tuners.append((name, tuner))

        logger.info("Tuning sequence completed.")
        return self.tuners

def generic_tuner(
    config,
    # builder,
    data,
    **kwargs,
):
    """A generic tuner that can tune any hyperparameter specified in kwargs.
    kwargs should be a dictionary where keys are the hyperparameter names,
    corresponding to config fields and values are lists of possible values.

    The modified config file is attached to the compiled model as model._config
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
    optimizers=["keras.optimizers.adam", "keras.optimizers.adamw", ],
):
    """Tune the optimizer and learning rate."""

    def build_model(hp):
        optimizer = hp.Choice("optimizer", optimizers)
        new_conf = config.copy()
        OmegaConf.update(new_conf, "optimizer.function", optimizer)
        builder = ModelBuilder(new_conf)
        return builder.compile()

    return build_model


