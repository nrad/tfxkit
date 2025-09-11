import keras_tuner as kt
import numpy as np
import logging
from omegaconf import OmegaConf
from tfxkit.common.base_utils import import_function
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
            tuner = kt.BayesianOptimization(
                model_builder,
                objective="val_loss",
                directory=name,
                project_name=f"HPTuning",
                max_trials=10,
                num_initial_points=None,
                alpha=0.0001,
                beta=2.6,
                seed=None,
                overwrite=True,
            )
            tuner.search(
                self.data.X_train,
                self.data.y_train,
                validation_split=0.2,
                sample_weight=self.data.sample_weight_train,
                batch_size=50_000,
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


from tfxkit.core.model_builder import ModelBuilder


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


# def progressive_layer_tuning(
#     config, builder, data,
#     units_choices=[], max_layers=5, patience=1, max_trials=20
# ):
#     """ Progressively tune the number of layers and units in each layer.
#     """

#     best_layers = []
#     best_val_loss = np.inf
#     no_improve_counter = 0
#     tuners = {}
#     builders = []

#     def make_build_model(n_layers, best_layers, builder, units_choices):
#         def build_model(hp):
#             layers = best_layers.copy()
#             new_units = hp.Choice(f"units_layers_{n_layers}", units_choices)
#             layers.append(new_units)
#             model_kwargs = dict(layers_list=layers)
#             model = builder.define_model(**model_kwargs)
#             return builder.compile(model)
#         return build_model

#     for n_layers in range(1, max_layers + 1):

#         build_model = make_build_model(n_layers, best_layers, builder, units_choices)
#         builders.append(build_model)


# def _progressive_layer_tuning(
#     config, builder, data,
#     units_choices=[], max_layers=5, patience=1, max_trials=20
# ):
#     """ Progressively tune the number of layers and units in each layer.
#     """

#     best_layers = []
#     best_val_loss = np.inf
#     no_improve_counter = 0
#     tuners = {}

#     def make_build_model(n_layers, best_layers, builder, units_choices):
#         def build_model(hp):
#             layers = best_layers.copy()
#             new_units = hp.Choice(f"units_layers_{n_layers}", units_choices)
#             layers.append(new_units)
#             model_kwargs = dict(layers_list=layers)
#             model = builder.define_model(**model_kwargs)
#             return builder.compile(model)
#         return build_model

#     for n_layers in range(1, max_layers + 1):

#         # def build_model(hp):
#         #     layers = best_layers.copy()
#         #     new_units = hp.Choice(f"units_layers_{n_layers}", units_choices)
#         #     layers.append(new_units)
#         #     model_kwargs = dict(layers_list=layers)
#         #     model = builder.define_model(**model_kwargs)
#         #     return builder.compile(model)
#         build_model = make_build_model(n_layers, best_layers, builder, units_choices)

#         tuner = kt.BayesianOptimization(
#             build_model,
#             objective="val_loss",
#             max_trials=max_trials,
#             directory="progressive_tune",
#             project_name=f"layers_{n_layers}",
#         )
#         tuners[n_layers] = (tuner, build_model)

#         tuner.search(
#             data.X_train,
#             data.y_train,
#             validation_split=config.training.validation_split,
#             sample_weight=data.sample_weight_train,
#             # verbose=0,
#         )

#         best_hp = tuner.get_best_hyperparameters(1)[0]
#         val_loss = tuner.oracle.get_best_trials(1)[0].score


#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             print(best_hp)
#             print(best_hp.values)

#             best_layers.append(best_hp.get(f"units_layers_{n_layers}"))
#             no_improve_counter = 0
#         else:
#             no_improve_counter += 1
#             if no_improve_counter >= patience:
#                 break

#     return best_layers, tuners


#     # tuner = kt.BayesianOptimization(
#     #     build_model,
#     #     objective="val_loss",
#     #     max_trials=max_trials,
#     #     directory="optimizer_tune",
#     #     project_name="optimizer",
#     # )

#     # tuner.search(
#     #     data.X_train,
#     #     data.y_train,
#     #     validation_split=config.training.validation_split,
#     #     sample_weight=data.sample_weight_train,
#     #     # verbose=0,
#     # )

#     # best_hp = tuner.get_best_hyperparameters(1)[0]
#     # best_optimizer = best_hp.get("optimizer")
#     # best_learning_rate = best_hp.get("learning_rate")

#     # return (best_optimizer, best_learning_rate), tuner

# # class LayerTuner(Tuner):
# #     def __init__(self, *args, **kwargs):
# #         super().__init__(*args, **kwargs)

# #     def progressive_layer_tuning(
# #         self, units_choices=[], max_layers=5, patience=1, max_trials=20
# #     ):
# #         # cfg = self.config.tuner.progressive_layer
# #         # max_layers = cfg.max_layers
# #         # patience = cfg.patience
# #         # max_trials = cfg.max_trials
# #         # new_units_choices = cfg.units_choices

# #         best_layers = []
# #         best_val_loss = np.inf
# #         no_improve_counter = 0
# #         tuners = {}

# #         for n_layers in range(1, max_layers + 1):

# #             def build_model(hp):
# #                 layers = best_layers.copy()
# #                 new_units = hp.Choice(f"units_layers_{n_layers}", units_choices)
# #                 layers.append(new_units)
# #                 model_kwargs = dict(layers_list=layers)
# #                 model = self.builder.define_model(**model_kwargs)
# #                 return self.builder.compile(model)

# #             tuner = kt.BayesianOptimization(
# #                 build_model,
# #                 objective="val_loss",
# #                 max_trials=max_trials,
# #                 directory="progressive_tune",
# #                 project_name=f"layers_{n_layers}",
# #             )
# #             tuners[n_layers] = tuner

# #             tuner.search(
# #                 self.data.X_train,
# #                 self.data.y_train,
# #                 validation_split=0.2,
# #                 sample_weight=self.data.sample_weight_train,
# #                 verbose=0,
# #             )

# #             best_hp = tuner.get_best_hyperparameters(1)[0]
# #             val_loss = tuner.oracle.get_best_trials(1)[0].score

# #             if val_loss < best_val_loss:
# #                 best_val_loss = val_loss
# #                 best_layers.append(best_hp.get(f"units_layers_{n_layers}"))
# #                 no_improve_counter = 0
# #             else:
# #                 no_improve_counter += 1
# #                 if no_improve_counter >= patience:
# #                     break

# #         return best_layers, tuners
