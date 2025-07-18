import tensorflow as tf

# from tensorflow.keras import layers, models
from tfxkit.common.base_utils import import_function
import logging
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)


class ModelBuilder:
    """
    Responsible for building and compiling the Keras model using config parameters.
    """

    def __init__(self, config):
        self.config = config
        self.model_config = self.config.model
        self.model_parameters = OmegaConf.to_container(
            self.model_config.parameters, resolve=True
        )
        self.optimizer_config = self.config.optimizer
        self.optimizer_parameters = OmegaConf.to_container(
            self.optimizer_config.parameters, resolve=True
        )

        self.loss = self.optimizer_config.loss
        self.metrics = list(self.optimizer_config.metrics)

        self.data_config = self.config.data
        self.features = self.data_config.features
        self.n_features = len(self.features)
        self.labels = self.data_config.labels
        self.n_labels = len(self.labels)
        self.model = self.define_model()
        

    def _load_model_fn(self):

        fn_path = self.model_config.get("function", "tfxkit.common.tf_utils.define_mlp")
        model_definer = import_function(fn_path)
        setattr(self, "definer", model_definer)
        return model_definer

    def define_model(self, **kwargs):
        self._load_model_fn()
        model_kwargs = self.model_parameters
        if kwargs:
            logger.debug(f"Updating model parameters with: {kwargs}")
            model_kwargs.update(kwargs)
        model_kwargs.update(
            {
                "n_features": self.n_features,
                "n_labels": self.n_labels,
            }
        )
        logger.info(
            f"Defining model with: {self.definer.__name__} and args: {model_kwargs}"
        )
        model = self.definer(**model_kwargs)
        
        return model

    def compile(self, model=None, loss=None, metrics=None, **kwargs):

        model = self.model if model is None else model
        loss = self.loss if loss is None else loss
        metrics = self.metrics if metrics is None else metrics

        optimizer_kwargs = self.optimizer_parameters
        optimizer_fn_name = self.optimizer_config.function

        if kwargs:
            logger.debug(f"Updating optimizer parameters with: {kwargs}")
            optimizer_kwargs.update(kwargs)
        logger.info(f"Compiling model with optimizer: {optimizer_fn_name} and parameters: \n{optimizer_kwargs}")

        if "." in optimizer_fn_name:
            optimizer_fn = import_function(optimizer_fn_name)
            optimizer_fn = optimizer_fn(**optimizer_kwargs)
        else:
            optimizer_fn = optimizer_fn_name

        logger.info(
            f"Compiling model with optimizer={optimizer_fn}, loss={loss}, metrics={metrics}"
        )
        model.compile(optimizer=optimizer_fn, loss=loss, metrics=metrics)
        return model
