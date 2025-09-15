import logging
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from sklearn.metrics import roc_curve, auc
from tfxkit.common import plotting_utils as pu
from pydoc import locate
import os

logger = logging.getLogger(__name__)


class Plotter:
    def __init__(self, config, data_manager, evaluator, trainer):
        self.config = config
        self.evaluator = evaluator
        self.data = data_manager
        self.trainer = trainer
        self.plot_config = self.config.get("plotter", {})
        self.plots_path = self.plot_config.get("plots_path", None)

    def locate_function(self, func_path):
        if hasattr(self, func_path):
            return getattr(self, func_path)
        else:
            func = locate(func_path)
            if func is None:
                raise ImportError(f"Could not locate function '{func_path}'")
            return func

    def resolve_plot_path(self, plot_name, plot_path=None):
        plot_path = plot_path if plot_path else self.plots_path
        if not plot_path:
            return None
        if plot_name:
            plot_path = os.path.join(plot_path, plot_name)
        return plot_path

    def plot_history(self, plot_path=None, **kwargs):
        # plot_history(history, ylim=None, xlabel="Epoch", ylabel="", plot_kwargs={}, keys=None):
        history = self.trainer.history
        plot_kwargs = dict(xlabel="Epoch", ylabel="Value")
        plot_kwargs.update(**kwargs)

        plot_path = self.resolve_plot_path("training_history", plot_path)
        plot_kwargs.update(dict(plot_path=plot_path))
        output = pu.plot_history(history, **plot_kwargs)
        return output

    def plot_predictions(
        self,
        # df1, df2
        variable="pred",
        x_label="Predictions",
        bins=50,
        range=(0, 1),
        plot_name=None,
        plot_path=None,
        weight_column=None,
        weight_column_train=None,
        **kwargs,
    ):

        # weight_column = self.data.data_config.get("weights_column", None)
        # weight_column_train =
        plot_path = plot_path if plot_path else self.plots_path
        if plot_path:
            plot_name = plot_name if plot_name else variable
            plot_path = os.path.join(plot_path, plot_name)
        target_label = self.data.data_config["labels"][0]
        output = pu.plot_classwise_hist(
            self.data.df_test,
            df_train=self.data.df_train,
            variable=variable,
            label_column=target_label,
            weight_column=weight_column,
            weight_column_train=weight_column_train,
            # weight_column=weights if weights is not None else self.data.data_config.get("weights_column", None),
            # weight_column_train=weights if weights is not None else self.data.data_config.get("weights_column", None),
            bins=bins,
            range=range,
            comparison="pull",
            plot_path=plot_path,
            **kwargs,
        )
        return output

    def run_sequence(self):
        sequence = self.plot_config.get("sequence", [])
        outputs = []
        for name in sequence:
            func_config = self.plot_config.functions[name]
            func_path = func_config.function
            logging.debug(f"Importing function: {func_path}")
            func = self.locate_function(func_path)
            if func is None:
                raise ImportError(f"Could not locate function '{func_path}'")

            kwargs = (
                OmegaConf.to_container(func_config.get("parameters"), resolve=True)
                if func_config.get("parameters")
                else {}
            )
            settings = (
                OmegaConf.to_container(func_config.get("settings"), resolve=True)
                if func_config.get("settings")
                else {}
            )

            logger.info(f"Running plotter func: {name}: {func_path}")

            plot_output = func(**kwargs, **settings)
            outputs.append((name, plot_output))
        return outputs
