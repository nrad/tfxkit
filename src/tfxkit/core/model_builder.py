import tensorflow as tf
from tensorflow import keras


from tfxkit.common.base_utils import import_function
from omegaconf import OmegaConf
import os
import logging

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

        self.save_dir = self.config.save_dir
        self.model_path = os.path.join(self.save_dir, "model.keras")

        if os.path.isfile(self.model_path) and self.model_config.reload_model:
            self.load_model()
        else:
            self.model = self.define_model()

    def _check_model_consistency_with_config(self):
        """
            TO DO!
            need a way to check that model loaded with self.load_model is consistent
            with the model one would get by building from the config file.
        """

    def _locate_model_function(self):

        fn_path = self.model_config.get("function", "tfxkit.common.tf_utils.define_mlp")
        model_definer = import_function(fn_path)
        setattr(self, "definer", model_definer)
        return model_definer

    def define_model(self, **kwargs):
        self._locate_model_function()
        model_kwargs = dict(**self.model_parameters)
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

    def summary(self):
        """Prints the model summary."""
        if self.model is not None:
            logger.info("Model Summary:")
            self.model.summary(print_fn=logger.info)
        else:
            logger.warning("Model is not defined yet. Cannot print summary.")

    def compile(self, model=None, loss=None, metrics=None, **kwargs):

        model = self.model if model is None else model
        loss = self.loss if loss is None else loss
        metrics = self.metrics if metrics is None else metrics

        optimizer_kwargs = dict(**self.optimizer_parameters)
        optimizer_fn_name = self.optimizer_config.function

        if kwargs:
            logger.debug(f"Updating optimizer parameters with: {kwargs}")
            optimizer_kwargs.update(kwargs)
        logger.info(
            f"Compiling model with optimizer: {optimizer_fn_name} and parameters: \n{optimizer_kwargs}"
        )

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


    def save_model(self, path, overwrite=None):

        path = path if path else self.model_path

        if self.model is None:
            logger.warning("Model is not defined yet. Cannot save model.")
            return

        # Check if the model has been compiled
        if not hasattr(self.model, "optimizer") or self.model.optimizer is None:
            logger.warning("Model has not been compiled or trained yet.")
            return

        if int(self.model.optimizer.iterations) == 0:
            logger.warning("Model has been compiled but not trained yet (no optimizer steps).")

        overwrite = overwrite if overwrite is not None else self.model_config.overwrite

        logger.debug(f"Saving model to {path} with overwrite={overwrite}")
        if os.path.isfile(path) and not overwrite:
                logger.warning(f"Model file {path} already exists and overwrite is set to False. Not saving.")
                raise FileExistsError(f"File {path} already exists and overwrite is set to {overwrite}")
            
        self.model.save(path, overwrite=overwrite)
        logger.info(f"Model saved successfully to {path}")    

    def load_model(self, path=None, compile=False):
        """
        Loads a saved Keras model from disk and assigns it to self.model.
        
        Args:
            path (str or Path): Path to the saved model directory or file (.h5, .keras, or SavedModel).
            compile (bool): Whether to compile the model after loading.
        """
        path = path if path else self.model_path
        try:
            logger.info(f"Loading model from {path} (compile={compile})...")
            self.model = keras.models.load_model(path, compile=compile)
            logger.debug("Model loaded successfully.")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            raise