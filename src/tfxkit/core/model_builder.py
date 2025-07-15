import tensorflow as tf

# from tensorflow.keras import layers, models
import importlib
import logging

logger = logging.getLogger(__name__)


class ModelBuilder:
    """
    Responsible for building and compiling the Keras model using config parameters.
    """

    def __init__(self, config):
        self.config = config
        self.model_config = self.config.get("model", {})
        self.model_parameters = self.model_config.get("parameters", {})
        self._load_model_fn()

    def _load_model_fn(self):
        fn_path = self.model_config.get("function", "tfxkit.common.tf_utils.define_mlp")
        module_path, fn_name = fn_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        if not hasattr(module, fn_name):
            raise ValueError(f"Function {fn_name} not found in module {module_path}")
        model_definer = getattr(module, fn_name)
        if not callable(model_definer):
            raise ValueError(f"{fn_name} is not a callable function")
        logger.info(f"Loaded model function {fn_name} from {module_path}")
        setattr(self, "definer", model_definer)
        return model_definer

    def define_model(self):
        model_kwargs = self.model_parameters
        logger.info(
            f"Building model with: {self.definer.__name__} and args: {model_kwargs}"
        )
        return self.definer(**model_kwargs)
