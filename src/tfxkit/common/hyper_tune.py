from tfxkit.common import base_utils, tf_utils
import os
import time


import keras
import keras_tuner as kt



class TimeStopping(keras.callbacks.Callback):
    """
    class back for monitoring the epoch duration
    """

    def __init__(self, max_seconds, verbose=0):
        super(TimeStopping, self).__init__()
        self.max_seconds = max_seconds
        self.verbose = verbose
        self.start_time = 0

    # def on_train_begin(self, logs=None):
    #     self.start_time = time.time()
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()


    def on_epoch_end(self, epoch, logs=None):
        if time.time() - self.start_time > self.max_seconds:
            self.model.stop_training = True
            if self.verbose:
                print(
                    f"\nStopping training after {epoch+1} epochs due to time limit of {self.max_seconds} seconds."
                )

class ModelSizeMonitor(keras.callbacks.Callback):
    def __init__(self, min_layers=5, max_layers=30, max_params=500_000, verbose=0):
        super(ModelSizeMonitor, self).__init__()
        self.max_layers = max_layers
        self.min_layers = min_layers
        self.max_params = max_params
        self.verbose = verbose

    def on_train_begin(self, epoch, logs=None):
        # Check number of layers
        if self.max_layers is not None and len(self.model.layers) > self.max_layers:
            if self.verbose:
                print(f"Stopping training: number of layers exceeded {self.max_layers}")
            self.model.stop_training = True

        if self.min_layers is not None and len(self.model.layers) < self.min_layers:
            if self.verbose:
                print(f"Stopping training: number of layers not enough {self.min_layers}")
            self.model.stop_training = True


        # Check number of parameters
        if self.max_params is not None and self.model.count_params() > self.max_params:
            if self.verbose:
                print(
                    f"Stopping training: number of parameters exceeded {self.max_params}"
                )
            self.model.stop_training = True


def model_builder_wrapper(mf, model_modifier=None, metrics=["accuracy"]):
    """
    Wrapper function for building a model using hyperparameters.
    
    Args:
        mf: The ModelFactory object.
        model_modifier: function that takes the model and returns a modified model.
    Returns:
        The model builder function.
    """
    
    def model_builder(hp):
        # Define hyperparameters to tune
        hyper_params = dict(
            # learning_rate = hp.Choice('learning_rate', values=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5] ),
            #init_layer=hp.Int("init_layer", min_value=4, max_value=1024, step=32),
            init_layer=hp.Choice("init_layer", values=[4, 8, 16, 32, 64, 96, 128, 192, 256, 320, 416, 512, 608, 704, 864, 1024]),
            final_layer=hp.Choice("final_layer", values=[2, 4, 8]),
            layer_repeat=hp.Int("layer_repeat", min_value=1, max_value=5, step=1),
            unit_step=hp.Choice("unit_step", values=[1, 2, 4, 8]),
            optimizer_name=hp.Choice(
                "optimizer_name",
                values=[
                    "adam",
                    "adamw",
                    "nadam",
                ],
            ),
            hidden_activation=hp.Choice(
                "hidden_activation", values=["relu", "tanh", "sigmoid", "leaky_relu"]
            ),
            learning_rate=hp.Choice("learning_rate", values=list(tf_utils.lr_dict)),
            weight_decay=hp.Choice(
                # "weight_decay", min_value=0.0001, max_value=0.001, step=0.0001
                "weight_decay", values=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            ),
            clipnorm=hp.Choice("clipnorm", values=[0.0, 0.5, 1.0, 5.0, 10.0]),
            use_ema=hp.Boolean("use_ema"),
        )

        ema_momentum = hp.Choice(
            "ema_momentum",
            values=[0.9, 0.99, 0.999],
            parent_name="use_ema",
            parent_values=[True],
        )
        hyper_params.update(ema_momentum=ema_momentum)

        with hp.conditional_scope("optimizer_name", ["adam", "adamw", "nadam"]):
            if hyper_params["optimizer_name"] == "adam":
                optimizer = keras.optimizers.Adam
            elif hyper_params["optimizer_name"] == "adamw":
                optimizer = keras.optimizers.AdamW
            elif hyper_params["optimizer_name"] == "nadam":
                optimizer = keras.optimizers.Nadam
            else:
                raise ValueError(f"{hyper_params['optimizer_name'] = } not recognized!")

            optimizer_params = utils.filter_kwargs(optimizer, hyper_params)
            print(hyper_params["optimizer_name"], optimizer_params)
            # optimizer = optimizer(**optimizer_params)

        model = mf.define_model(**hyper_params)
        if model_modifier is not None:
            model = model_modifier(model)
        # model.summary()
        print(
            f"====== model info: \nn_params: {model.count_params()}, \nn_layers: {len(model.layers)}, \nlayers: {[x.name for x in model.layers]}"
        )
        mf.compile_model(
            optimizer, optimizer_kwargs=optimizer_params, metrics=metrics
        )

        return model

    return model_builder


def evaluate_tuner(tuner, X, y, batch_size=8000, num_trials=10):
    import time

    res = []
    models = tuner.get_best_models(num_trials)
    hps = tuner.get_best_hyperparameters(num_trials=num_trials)
    res = []
    for i, m in enumerate(models):
        start_time = time.time()
        di = m.evaluate(
            #X, y, batch_size=batch_size, use_multiprocessing=True, return_dict=True
            X, y, batch_size=batch_size, return_dict=True
        )
        eval_time = round(time.time() - start_time, 3)
        hp = hps[i]
        res.append(
            dict(model=m, hp=hp, eval_time=eval_time, **di)
        )
    return res


# res = [ (m, hp, m.evaluate(mf.X_test, mf.y_test, batch_size=mf.batch_size, use_multiprocessing=True, return_dict=True)) for i,m in enumerate(ms)]


if __name__ == "__main__":
    model_name = "model_36354d32"

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_name = args.model_name

    mf = ModelFactory.load_model(model_name)
    # mf._df_train = mf.df_train.sample(100_000)

    tuner = kt.Hyperband(
        model_builder_wrapper(mf),
        objective="loss",
        max_epochs=20,
        factor=3,
        directory=os.path.join(mf.model_dir, "keras_tuner"),
        project_name="hyperband",
        overwrite=args.overwrite,
        # batch_size=1000,
    )

    tuner = kt.BayesianOptimization(
        model_builder_wrapper(mf),
        # objective='binary_crossentropy',
        objective="loss",
        # max_epochs=20,
        # factor=3,
        directory=os.path.join(mf.model_dir, "keras_tuner"),
        project_name="bayesianoptimization",
        max_trials=20,
        num_initial_points=None,
        alpha=0.0001,
        beta=2.6,
        seed=None,
        overwrite=args.overwrite,
        # hyperparameters=None,
        # tune_new_entries=True,
        # allow_new_entries=True,
        # max_retries_per_trial=0,
        # max_consecutive_failed_trials=3,
        # **kwargs
    )

    # assert False

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="loss",  # Monitor validation loss
        min_delta=0.001,  # Minimum change to qualify as an improvement
        patience=1,  # How many epochs to wait for improvement
        mode="auto",  # Direction of improvement is auto-inferred
        restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
        verbose=1,
    )

    time_stopping_callback = TimeStopping(max_seconds=120, verbose=1)

    tuner.search(
        mf.X_train,
        mf.y_train,
        epochs=10,
        batch_size=20000,
        callbacks=[early_stopping_callback, time_stopping_callback],
        validation_split=0.2,
    )
