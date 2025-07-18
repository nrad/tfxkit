from tfxkit.core.data_manager import DataManager
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Handles training the model, including callbacks and checkpoints.
    """

    def __init__(self, config, model, data_manager: DataManager):
        self.config = config
        self.training_config = self.config.training
        self.eval_config = self.config.evaluation
        self.model = model
        self.data = data_manager

    def add_test_train_preds(self, pred_key="pred", **kwargs):
        df_train = self.data.df_train
        df_test = self.data.df_test
        X_train = self.data.X_train
        X_test = self.data.X_test

        predict_kwargs = dict(
            batch_size=self.eval_config.get(
                "batch_size", self.training_config.batch_size
            ),
            verbose=1,
        )
        df_train["pred"] = self.predict(x=X_train, **predict_kwargs)
        df_test["pred"] = self.predict(x=X_test, **predict_kwargs)

    def predict(self, **kwargs):
        """wrapper for model.predict"""
        predictions = self.model.predict(**kwargs)
        return predictions

    def evaluatee(self, **fit_kwargs):
        """Train the model using the configured parameters and data."""
        # self.data.load_df()
        pass
