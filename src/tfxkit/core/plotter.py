# from tfxkit.common.base_utils import logger
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from sklearn.metrics import roc_curve, auc
from tfxkit.common import plotting_utils as pu
from pydoc import locate
import os
import logging

logger = logging.getLogger(__name__)

class Plotter:

    def __init__(self, config, data_manager, evaluator, trainer):
        self.config = config
        self.evaluator = evaluator
        self.data = data_manager
        self.trainer = trainer
        self.plot_config = self.config.get("plotter", {})
        self.plots_path = self.plot_config.get(
            "plots_path", self.config.get("save_dir", None)
        )

    def locate_function(self, func_path):
        """
        Locate and return a function given its full path as a string.
        If the function is a method of the class, it retrieves it directly.
        Raises ImportError if the function cannot be found.

        Args:
            func_path (str): The full path to the function, e.g., 'module.sub
            module.function_name' or 'ClassName.method_name'.
        Returns:
            function: The located function object.
        Raises:
            ImportError: If the function cannot be located.
        """

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

    def save_fig(self, fig, plot_name=None, plot_path=None, **kwargs):
        plot_path = self.resolve_plot_path(plot_name, plot_path)
        if not plot_path:
            logger.warning("No plot path specified. Figure will not be saved.")
            return None
        pu.save_fig(fig, plot_path, **kwargs)
        return plot_path

    def plot_history(self, plot_path=None, **kwargs):
        # plot_history(history, ylim=None, xlabel="Epoch", ylabel="", plot_kwargs={}, keys=None):
        if not hasattr(self.trainer, "history"):
            logger.warning("No training history found to plot.")
            return None
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
        bins=50,
        range=(0, 1),
        plot_name=None,
        plot_path=None,
        weight_column=None,
        weight_column_train=None,
        **kwargs,
    ):

        plot_path = self.resolve_plot_path(plot_name, plot_path)

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

    def plot_roc(
        self,
        test_train=["test", "train"],
        variable=None,
        subplots_kwargs={},
        test_kwargs={},
        train_kwargs={},
        plot_path=None,
    ):
        variable = self.config.evaluation.get("variable")
        if not variable:
            raise ValueError(
                "No variable specified for ROC plot in evaluation config nor in function call."
            )

        target_label = self.config.data.labels
        if len(target_label) != 1:
            raise ValueError("ROC plotter only supports single label classification.")
        target_label = target_label[0]

        test_train = [test_train] if isinstance(test_train, str) else test_train

        fig_ax = plt.subplots(**subplots_kwargs)

        output = {}

        # Merge default kwargs with user-provided test_kwargs and train_kwargs
        kwargs = {
            "test": dict(
                label="Test",
                color="C0",
            ),
            "train": dict(label="Train", color="C1", linestyle="--"),
        }
        # Update defaults with any provided kwargs
        if test_kwargs:
            kwargs["test"].update(test_kwargs)
        if train_kwargs:
            kwargs["train"].update(train_kwargs)

        for tt in test_train:
            if tt not in ["test", "train"]:
                raise ValueError(
                    f"Invalid value '{tt}' for test_train argument. Must be 'test', 'train' or a list of these."
                )

            df = getattr(self.data, tt).df

            output[tt] = pu.plot_roc(
                fig_ax=fig_ax,
                truth=df[target_label],
                pred=df[variable],
                **kwargs[tt],
            )
        if not plot_path is False:
            plot_path = self.resolve_plot_path("roc_curve", plot_path)
            self.save_fig(fig_ax[0], plot_path)
        return output

    def run_sequence(self):
        sequence = self.plot_config.get("sequence", [])
        outputs = []
        for name in sequence:
            if name not in self.plot_config.functions:
                logger.debug(
                    f"Plotter function '{name}' not found in config.plotter.functions. \
                        Will assume it's a method of the Plotter class."
                )
                func_path = name
                kwargs = {}

            else:
                func_config = self.plot_config.functions[name]
                func_path = func_config.function
                logger.debug(f"Importing function: {func_path}")
                kwargs = (
                    OmegaConf.to_container(func_config.get("parameters"), resolve=True)
                    if func_config.get("parameters")
                    else {}
                )

            func = self.locate_function(func_path)
            if func is None:
                raise ImportError(f"Could not locate function '{func_path}'")

            logger.info(f"Running plotter func: {name}: {func_path}")

            plot_output = func(**kwargs)
            outputs.append((name, plot_output))
        return outputs
