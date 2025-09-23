from tfxkit.core.data_manager import DataManager
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class Trainer:
    """
    Handles training the model, including callbacks and checkpoints.
    """

    def __init__(self, config, model, data_manager: DataManager):
        self.config = config
        self.training_config = self.config.training
        self.model = model
        self.data = data_manager
        # self.

    def fit(self, **kwargs):
        """
        Low-level training method.
        This directly calls `model.fit(...)` on the underlying Keras model.
        It does **not** automatic saving so normally you should call 
        `ModelFactory.fit()` instead.
        """
    
        # self.data.load_df()

        fit_kwargs = dict(
            # batch_size=self.training_config.batch_size,
            # epochs=self.training_config.epochs,
            verbose=1,
            x=self.data.X_train,
            y=self.data.y_train,
            sample_weight=self.data.sample_weight_train,
        )
        fit_kwargs.update(self.training_config)
        if kwargs:
            fit_kwargs.update(kwargs)
            logger.debug(f"Updating kwargs with config: {fit_kwargs}")
        logger.info(f"Training model with fit_kwargs: {fit_kwargs}")
        self.history = self.model.fit(**fit_kwargs)
        # self.fit_kwargs = fit_kwargs
        return self.history
