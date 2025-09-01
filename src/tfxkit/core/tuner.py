import keras_tuner as kt
import numpy as np
import logging
from omegaconf import OmegaConf
from tfxkit.common.base_utils import import_function
from pydoc import locate

logger = logging.getLogger(__name__)


class Tuner:
    def __init__(self, config, builder, data):
        self.config = config
        self.builder = builder
        self.data = data

    def run_sequence(self):
        for name in self.config.tuner.sequence:
            func_config = self.config.tuner.functions[name]
            func_path = func_config.function
            logging.debug(f"Importing function: {func_path}")
            print(f"Importing function: {func_path}")
            func = locate(func_path)
            if func is None:
                raise ImportError(f"Could not locate function '{func_path}'")

            kwargs = OmegaConf.to_container(
                func_config.get("parameters", {}), resolve=True
            )
            settings = OmegaConf.to_container(
                func_config.get("settings", {}), resolve=True
            )
            logger.info(f"Running tuner step: {name}")
            # func(self, **kwargs, **settings)
            func(config=self.config, builder=self.builder, data=self.data, **kwargs, **settings)

        logger.info("Tuning sequence completed.")

def progressive_layer_tuning(
    config, builder, data, 
    units_choices=[], max_layers=5, patience=1, max_trials=20
):
    # cfg = self.config.tuner.progressive_layer
    # max_layers = cfg.max_layers
    # patience = cfg.patience
    # max_trials = cfg.max_trials
    # new_units_choices = cfg.units_choices

    best_layers = []
    best_val_loss = np.inf
    no_improve_counter = 0
    tuners = {}

    for n_layers in range(1, max_layers + 1):

        def build_model(hp):
            layers = best_layers.copy()
            new_units = hp.Choice(f"units_layers_{n_layers}", units_choices)
            layers.append(new_units)
            model_kwargs = dict(layers_list=layers)
            model = builder.define_model(**model_kwargs)
            return builder.compile(model)

        tuner = kt.BayesianOptimization(
            build_model,
            objective="val_loss",
            max_trials=max_trials,
            directory="progressive_tune",
            project_name=f"layers_{n_layers}",
        )
        tuners[n_layers] = tuner

        tuner.search(
            data.X_train,
            data.y_train,
            validation_split=0.2,
            sample_weight=data.sample_weight_train,
            verbose=0,
        )

        best_hp = tuner.get_best_hyperparameters(1)[0]
        val_loss = tuner.oracle.get_best_trials(1)[0].score

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_layers.append(best_hp.get(f"units_layers_{n_layers}"))
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            if no_improve_counter >= patience:
                break

    return best_layers, tuners



# class LayerTuner(Tuner):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def progressive_layer_tuning(
#         self, units_choices=[], max_layers=5, patience=1, max_trials=20
#     ):
#         # cfg = self.config.tuner.progressive_layer
#         # max_layers = cfg.max_layers
#         # patience = cfg.patience
#         # max_trials = cfg.max_trials
#         # new_units_choices = cfg.units_choices

#         best_layers = []
#         best_val_loss = np.inf
#         no_improve_counter = 0
#         tuners = {}

#         for n_layers in range(1, max_layers + 1):

#             def build_model(hp):
#                 layers = best_layers.copy()
#                 new_units = hp.Choice(f"units_layers_{n_layers}", units_choices)
#                 layers.append(new_units)
#                 model_kwargs = dict(layers_list=layers)
#                 model = self.builder.define_model(**model_kwargs)
#                 return self.builder.compile(model)

#             tuner = kt.BayesianOptimization(
#                 build_model,
#                 objective="val_loss",
#                 max_trials=max_trials,
#                 directory="progressive_tune",
#                 project_name=f"layers_{n_layers}",
#             )
#             tuners[n_layers] = tuner

#             tuner.search(
#                 self.data.X_train,
#                 self.data.y_train,
#                 validation_split=0.2,
#                 sample_weight=self.data.sample_weight_train,
#                 verbose=0,
#             )

#             best_hp = tuner.get_best_hyperparameters(1)[0]
#             val_loss = tuner.oracle.get_best_trials(1)[0].score

#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_layers.append(best_hp.get(f"units_layers_{n_layers}"))
#                 no_improve_counter = 0
#             else:
#                 no_improve_counter += 1
#                 if no_improve_counter >= patience:
#                     break

#         return best_layers, tuners
