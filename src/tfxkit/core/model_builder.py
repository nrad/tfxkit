"""
ModelBuilder is a class that builds and compiles the Keras model using config parameters.
"""
import os
import logging
import keras
from omegaconf import OmegaConf
from tfxkit.common.base_utils import import_function, get_required_positional_arguments
from tfxkit.common.tf_utils import set_seeds
logger = logging.getLogger(__name__)


class ModelBuilder:
    """
    Responsible for building and compiling the Keras model using config parameters.
    """

    @property
    def model_parameters(self):
        return OmegaConf.to_container(
            self.model_config.parameters, resolve=True
        )

    @property
    def optimizer_parameters(self):
        return OmegaConf.to_container(
            self.optimizer_config.parameters, resolve=True
        )

    @property
    def metrics(self):
        return list(self.optimizer_config.metrics)

    @property
    def loss(self):
        return self.optimizer_config.loss

    

    def __init__(self, config):
        self.config = config
        self.model_config = self.config.model
        self.optimizer_config = self.config.optimizer

        self.data_config = self.config.data
        self.features = self.data_config.features
        self.n_features = len(self.features)
        self.labels = self.data_config.labels
        self.n_labels = len(self.labels)

        self.set_random_seed()

        model_debugger = self.model_config.get("debug", False)
        if model_debugger:
            return
        
        self.define_model()
        if os.path.isfile(self.model_path) and self.model_config.reload_model:
            logger.info(f"Loading model weights from {self.model_path}")
            try:
                self.model.load_weights(self.model_path)
            except Exception as e:
                logger.error(f"Failed to load model weights from {self.model_path}: {e}")
                raise e


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
    def model_path(self):
        return os.path.join(self.model_dir, "model.keras")


    def _check_model_consistency_with_config(self):
        """
            TO DO!
            need a way to check that model loaded with self.load_model is consistent
            with the model one would get by building from the config file.
        """

    # def _locate_model_function(self):

    @property
    def definer(self):
        """
        Returns the model function to be used to build the model, based on the 
        function path in the config. If the function is not found, it will 
        default to tfxkit.common.tf_utils.define_mlp.
        """
        fn_path = self.model_config.get("function", "tfxkit.common.tf_utils.define_mlp")
        model_definer = import_function(fn_path)
        return model_definer

    def define_model(self, attach_to_builder=True, **kwargs):
        """
        Builds the model using the model function returned by the definer property.
        """
        model_kwargs = dict(**self.model_parameters)
        if kwargs:
            logger.debug(f"Updating model parameters with: {kwargs}")
            model_kwargs.update(kwargs)
        model_kwargs.update(
            {
                # "n_features": self.n_features,
                "features": self.features,
                "n_labels": self.n_labels,
            }
        )

        required_pos_args = get_required_positional_arguments(self.definer)
        logger.info(f"Required positional arguments in {self.definer.__name__}: {required_pos_args}")
        extra_args = {'features': self.features, 'labels': self.labels, }
        for arg in required_pos_args:
            if arg not in model_kwargs:
                if arg in extra_args:
                    model_kwargs[arg] = extra_args[arg]
                else:
                    raise ValueError(f"Required positional argument {arg} not found in model parameters")

        logger.info(
            f"Defining model with: {self.definer.__name__} and args: {model_kwargs}"
        )
        model = self.definer(**model_kwargs)
        if attach_to_builder:
            self.model = model
        return model

    def set_random_seed(self, seed=None):
        seed = seed if seed is not None else self.model_config.get("random_seed", 999)
        logger.info(f"Setting random seed to {seed}")
        set_seeds(seed)

    def summary(self):
        """Prints the model summary."""
        if self.model is not None:
            logger.info("Model Summary:")
            self.model.summary(print_fn=logger.info)
        else:
            logger.warning("Model is not defined yet. Cannot print summary.")

    def compile(self, model=None, loss=None, metrics=None, **kwargs):
        """
        Compiles the model using the optimizer and loss function specified in the config.
        """

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

        if "." in loss:
            loss_fn = import_function(loss)
            logger.info(f"Imported loss function: {loss_fn.__name__}")
            loss_kwargs = self.optimizer_config.get("loss_kwargs", {})
            if loss_kwargs:
                logger.info(f"Updating loss function parameters with: {loss_kwargs}")
                # loss_fn = loss_fn.from_config(loss_kwargs)
                loss_fn = loss_fn(**loss_kwargs)
        else:
            loss_fn = loss


        logger.info(
            f"Compiling model with optimizer={optimizer_fn}, loss={loss_fn}, metrics={metrics}"
        )
        model.compile(optimizer=optimizer_fn, loss=loss_fn, metrics=metrics)
        return model

    # def build_model(self, model=None, batch_size=None, n_input_features=None):
    #     """
    #     Builds the model using the model function returned by the definer property.
    #     """
    #     model = self.model if model is None else model
    #     model.build(input_shape=(batch_size, n_input_features))
    #     return model

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
        os.makedirs(os.path.dirname(path), exist_ok=True)
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