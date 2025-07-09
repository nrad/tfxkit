import keras
from keras import layers
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import warnings
from packaging.version import Version


# from utils import

import utils
from utils import tf_utils

from features import select_features, DEFAULT_FEATURES

import inspect

# from copy import deepcopy

from defaults import default_config

# from utils.speedup_utils import model_modifier_logistic_activation
from utils.speedup_utils import custom_loss_dict, model_modifiers_dict


class ModelFactory:
    """
    ModelFactory is a class to handle the training, testing, and hyperparameter tuning of a keras model.
    The main purpose of this class is to provide a simple interface to train and test a keras model,
    and to save the model, its history, and the hyperparameters in a structured way.

    The input parameters are given as keyword arguments, and the class will use the default values if not given.
    The required input parameters are:
    - `fname_train`: a string or a list of strings, the file name(s) of the training dataset.
    - `fname_test`: a string or a list of strings, the file name(s) of the testing dataset.
    - `features`: a list of strings, the names of the input features.
    - `labels`: a list of strings, the names of the output labels.

    The class will automatically load the data from the given file names, and will split the data into training and testing datasets.
    It will also automatically define and compile the model, and will train the model using the training dataset.

    """

    default_dict = {
        "version_tag": default_config["version_tag"],
        "model_tag": default_config["model_tag"],
        "n_epoch": default_config["n_epoch"],
        "init_layer": default_config["init_layer"],
        "final_layer": default_config["final_layer"],
        "layer_repeat": default_config["layer_repeat"],
        "unit_step": default_config["unit_step"],
        "seed": default_config["seed"],
        "batch_size": default_config["batch_size"],
        # "chunk_size": default_config["chunk_size"],
        "optimizer": default_config["optimizer"],
        "learning_rate": default_config["learning_rate"],
        "dropout": default_config["dropout"],
        "hidden_activation": default_config["hidden_activation"],
        "model_dir_base": default_config["model_base_dir"],
        "config_dir_base": default_config["config_base_dir"],
        # "ema_momentum": default_config["ema_momentum"],
        "model_definer": default_config["model_definer"],
        "class_weight": None,
        # "model_"
        "weights": None,
        "validation_split": 0.2,
        "model_modifiers": [],
        "loss": default_config["loss"],
    }

    def __init__(self, **kwargs):
        di = {k: v for k, v in self.default_dict.items()}
        di.update(**kwargs)
        if not di.get("model_name"):
            di["model_name"] = utils.unique_name("model", 8)

        self.hyper_params = dict(**di)

        self.format_kwargs = {
            k: v for k, v in di.items() if k in ["model_tag", "version_tag"]
        }

        di["fname_train"] = utils.get_file_list(di["fname_train"], **self.format_kwargs)
        di["fname_test"] = utils.get_file_list(di["fname_test"], **self.format_kwargs)

        for k, v in di.items():
            setattr(self, k, v)

        if not hasattr(self, "plot_dir"):
            plot_dir_base = getattr(
                self, "plot_dir_base", default_config["plot_base_dir"]
            )
            self.plot_dir_base = plot_dir_base

            self.plot_dir = os.path.join(
                plot_dir_base,
                self.version_tag,
                self.model_tag if self.model_tag else "models",
                self.model_name,
            )
            # self.hyper_params.update(plot_dir=self.plot_dir)

        if not hasattr(self, "model_dir"):
            self.model_dir = os.path.join(
                self.model_dir_base, self.version_tag, self.model_tag, self.model_name
            )
        else:
            print(mf.model_tag)
            print(mf.model_dir)
            assert False, "model_dir is not implemented yet"
            # self.hyper_params.update(model_dir=self.model_dir)

        if Version(tf.__version__) > Version("2.17.0"):
            self.weights_path = os.path.join(
                self.model_dir, f"{self.model_name}.weights.h5"
            )
        else:
            self.weights_path = os.path.join(self.model_dir, f"weights.h5")

        required = ["fname_train", "fname_test", "features", "labels"]
        missing = [req for req in required if req not in kwargs]
        if len(missing):
            raise ValueError(f"Some required arguments are missing: {missing}")

        if hasattr(self, "seed"):
            tf_utils.set_seeds(self.seed)

        if len(self.fname_train) > 1:
            print(
                f"Multiple input files ({len(self.fname_train)})... will load them as TF Datasets:"
            )

            self.file_list_train, self.file_list_val, _ = utils.split_file_list(
                self.fname_train,
                val_frac=self.validation_split,
                test_frac=0.0,
            )
            if not len(self.file_list_val):
                raise ValueError(
                    f"validation_split ({self.validation_split}) probably too small for the given file list ({len(self.file_list_train)} files)"
                )
            if not len(self.file_list_train):
                raise ValueError(
                    f"file_list_train is empty! : {self.file_list_train = }"
                )

            print(
                f"""
            {self.file_list_train = } 
            {self.file_list_val   = }
            {self.fname_test      = }
                  """
            )
            self.ds_train = self.get_tf_dataset(self.file_list_train)
            self.ds_val = self.get_tf_dataset(self.file_list_val)
            self.ds_test = self.get_tf_dataset(self.fname_test)
        self.__add_train_properties()

        #
        print(self.model_definer)
        self.model_definer = getattr(
            tf_utils, self.model_definer, tf_utils.define_model
        )
        print(self.model_definer)
        # assert False

    @classmethod
    def __add_train_properties(cls):
        """
        dynamically add properties by caching the values
        using the appropriate functions
        """
        for attr_name in ["X_test", "X_train", "y_test", "y_train"]:

            def func(self, attr_name=attr_name):
                if not hasattr(self, f"_{attr_name}"):
                    self.prep_train_test()
                return getattr(self, f"_{attr_name}")

            setattr(cls, attr_name, property(func))

        for attr_name in ["df_test", "df_train"]:

            def func(self, attr_name=attr_name):
                if not hasattr(self, f"_{attr_name}"):
                    attr = self.load_df(
                        fname=getattr(self, attr_name.replace("df", "fname")),
                        sample_size=getattr(self, "sample_size", None),
                        selection=getattr(self, "selection", ""),
                        # selection="" if attr_name == "df_test" else getattr(self, "selection", ""),
                        n_files=getattr(self, attr_name.replace("df", "n_files"), 1),
                    )

                    setattr(self, f"_{attr_name}", attr)
                return getattr(self, f"_{attr_name}")

            setattr(cls, attr_name, property(func))

    @classmethod
    def load_model(
        cls,
        model_name,
        model_dir_base=default_config["model_base_dir"],
        version_tag=default_config["version_tag"],
        model_tag=default_config["model_tag"],
    ):
        # model_path = os.path.join(model_dir, "model.keras")
        model_dir = os.path.join(model_dir_base, version_tag, model_tag, model_name)
        model_path = os.path.join(model_dir, "model.keras")
        if os.path.isfile(model_path):
            print("loading model from: ", model_path)

            hyper_params = pickle.load(
                open(os.path.join(model_dir, "hyper_params.pkl"), "rb")
            )
            if "model_name" not in hyper_params:
                hyper_params["model_name"] = os.path.basename(model_name)
            mf = cls(**hyper_params)
            # model_extra_kwargs = dict(custom_objects={k:model_modifiers_dict[k] for k in mf.model_modifiers})
            # model_extra_kwargs = dict(custom_objects=dict(LogisticActivationLayer=utils.speedup_utils.LogisticActivationLayer))
            custom_objects = {}
            # custom_objects = dict(LogisticActivationLayer=utils.speedup_utils.LogisticActivationLayer)

            # model_extra_kwargs = {}
            # print(f"{model_extra_kwargs = }")
            # print("keras.models.load_model(model_path, custom_objects={'LogisticActivationLayer': utils.speedup_utils.LogisticActivationLayer})")
            # model = keras.models.load_model(model_path, custom_objects={'LogisticActivationLayer': utils.speedup_utils.LogisticActivationLayer})
            model = keras.models.load_model(model_path, custom_objects=custom_objects)

            print("MODEL", model)
            # load weights:
            weights_path = os.path.join(
                model_dir,
                "%sweights.h5"
                % (
                    "%s." % model_name
                    if Version(tf.__version__) > Version("2.17.0")
                    else ""
                ),
            )
            # weights_path = self.weights_path
            if os.path.isfile(weights_path):
                print("loading weights from: ", weights_path)
                model.load_weights(weights_path)
            else:
                print(f"could not find weights in {weights_path}")

            history = pickle.load(open(os.path.join(model_dir, "history.pkl"), "rb"))
            # return dict(model=model, history=history, hyper_params=hyper_params)
            mf.model = model
            mf.history = history
            mf.params = hyper_params
            # mf.model_name = os.path.basename(model_name)
            return mf
        else:
            raise ValueError(f"could not find model in path: {model_path}")

    @classmethod
    def load_config(
        cls,
        model_name_or_config_path,
        # model_dir_base=DEFAULT_MODEL_BASE_DIR,
        # version_tag=DEFAULT_VERSION_TAG,
        # model_tag=DEFAULT_MODEL_TAG,
        **extra_args,
    ):
        print(f"{extra_args = }")
        # print(f"{model_tag = }")

        # config_path = os.path.join(model_dir, "model.keras")
        if model_name_or_config_path.endswith(".yml"):
            config_path = model_name_or_config_path
        else:
            # model_name = model_name_or_config_path
            # model_dir = os.path.join(model_dir_base, version_tag, model_tag, model_name)
            # config_path = os.path.join(model_dir, "config.yml")
            raise NotImplementedError("only .yml config files are supported for now")

        if os.path.isfile(config_path):
            import yaml

            config = yaml.safe_load(open(config_path, "r"))

            for key in ["plot_dir", "model_dir", "config", "interactive"]:
                config.pop(key, None)
                extra_args.pop(key, None)

            # print("\n load config")
            if not extra_args:
                # print("not")
                if "model_name" in config:
                    try:
                        print("... Loading model from config['model_name']")
                        mf = cls.load_model(config["model_name"])
                        return mf
                    except ValueError:
                        pass
            else:  # extra arguments are passed
                if list(extra_args) != ["model_name"]:
                    # print("model_name in extra_args")
                    config_model_name = config.pop("model_name")
                    if "model_name" in config and "model_name" in extra_args:
                        if config_model_name == extra_args.get("model_name"):
                            print(
                                "WARNING: Since extra arguments are specified, a new model_name will be chosen!"
                            )
                            extra_args.pop("model_name")
                # elif "model_name" not in extra_args:
                #     print("config pop model_name")
                #     config.pop("model_name", None)
                add_features = extra_args.pop("add_features", [])
                remove_features = extra_args.pop("remove_features", [])
                features = extra_args.pop("features", config.get("features", []))
                if add_features or remove_features or features:
                    extra_args["features"] = select_features(
                        features, add_features, remove_features
                    )

            print("====== Updating config with extra args:")
            print(extra_args)
            # extra_args.setdefault("model_tag", model_tag)
            # extra_args.setdefault("version_tag", version_tag)
            config.update(**extra_args)
            # config.setdefault("model_tag", model_tag)
            # config.setdefault("version_tag", version_tag)

            print("config:")
            utils.pprint(config)
            print("... Initalizing Model from config")
            mf = cls(**config)
            return mf
        else:
            raise ValueError(f"could not find config in path: {config_path}")

    def save_config(self, config_path=None, clean_kwargs=False, **extra_kwargs):
        hyper_params = utils.deepcopy(self.hyper_params)

        if extra_kwargs:
            if not config_path:
                raise ValueError(
                    "<extra_kwargs> will modify the current hyper_params, so <config_path> should be given to avoid overwriting the current model's config"
                )
            if clean_kwargs:
                hyper_params = {
                    k: v
                    for k, v in hyper_params.items()
                    if k not in self.model_optimizer_kwargs
                }
                pass
            hyper_params.update(**extra_kwargs)

        config_path = (
            os.path.join(self.model_dir, "config.yml")
            if not config_path
            else config_path
        )

        utils.yaml_dump(hyper_params, fname=config_path)

    def save_model(self):
        import json

        # model_name = self.model
        model_dir = self.model_dir

        os.makedirs(model_dir, exist_ok=True)

        self.model.save(model_dir + "/model.keras")
        self.model_summary(file_path=f"{model_dir}/model_summary.txt")

        # self.model.save_weights(model_dir + f"/weights.h5")
        self.model.save_weights(self.weights_path)

        pickle.dump(self.history, open(f"{model_dir}/history.pkl", "wb"))
        pickle.dump(self.hyper_params, open(f"{model_dir}/hyper_params.pkl", "wb"))

        json.dump(
            self.hyper_params,
            open(os.path.join(self.plot_dir, "hp.json"), "w"),
            indent=4,
        )
        # model_summary = tf_utils.get_model_summary(self.model)
        # print(model_summary, file=open(os.path.join(self.plot_dir, "model.txt"), "w"))
        self.model_summary(html=True, file_path=f"{self.plot_dir}/model.html")

        self.save_config()
        self.save_config(config_path=os.path.join(self.plot_dir, "config.yml"))
        print("model saved in %s" % model_dir)

    def load_df(
        self,
        fname,
        sample_size=None,
        n_files=5,
        # where=None,
        selection="",
        sample_balancer=None,
        balancer_kwargs={},
    ):
        file_list = utils.get_file_list(fname, **self.format_kwargs)
        if n_files > 0:
            print(f"loading only {n_files} out of {len(file_list)}...")
            file_list = file_list[:n_files]
        print(f"loading DF: {fname}:\n       {file_list}")
        if len(file_list) == 0:
            raise ValueError(f"No file found at {fname}")

        df = pd.concat(
            # [utils.read_hdf(f, preselection=selection) for f in file_list]
            [utils.read_hdf(f, postselection=selection) for f in file_list]
        ).reset_index()
        if sample_balancer:
            _, df = tf_utils.balance_df(
                df,
                [k for k in df.columns if k not in self.labels],
                labels=self.labels,
                method=sample_balancer,
                **balancer_kwargs,
            )

        if sample_size:
            print("only using %s out %s rows:" % (sample_size, len(df)))
            df = df.sample(sample_size)
        # return df
        missing_columns = np.array([c for c in self.features if c not in df.columns])
        # found_columns = np.array([c in cols for c in self.features])
        # selection = getattr(self, "selection", "")
        if len(missing_columns):
            # missing_columns = df.columns[found_columns == False]
            print(
                "could not find these feature columns in the DF: %s\nWill run preproc"
                % missing_columns
            )
            preproc = getattr(self, "preproc", utils.preproc)

            df = preproc(df, query=selection)

            # df = utils.preproc(df, query=selection)
        else:
            if selection:
                df = df.query(selection)
        return df

    def split_df(self, df, frac=None, shuffle=True, seed=None):
        from sklearn.model_selection import train_test_split

        df_train, df_val = train_test_split(
            df,
            test_size=frac if frac else self.validation_split,
            random_state=seed if seed else self.seed,
            shuffle=shuffle,
        )
        return df_train, df_val

    def prep_train_test(self):
        # df_train = self.load_df(fname=self.fname_train)
        test_size = getattr(self, "test_size", None)

        if hasattr(self, "_X_train"):
            return self._X_train, self._X_test, self._y_train, self._y_test

        elif hasattr(self, "xy_maker"):
            Xy_train = self.xy_maker(self, self.df_train)
            Xy_test = self.xy_maker(self, self.df_test)
            X_train, X_test, y_train, y_test = (
                Xy_train["x"],
                Xy_test["x"],
                Xy_train["y"],
                Xy_test["y"],
            )
            self._X_train = X_train
            self._X_test = X_test
            self._y_train = y_train
            self._y_test = y_test

        else:
            X_train, X_test, y_train, y_test = tf_utils.prep_train_test(
                self.df_train,
                self.features,
                self.labels,
                test_size=test_size,
                df_test=self.df_test,
                seed=self.seed,
            )
            self._X_train = X_train
            self._X_test = X_test
            self._y_train = y_train
            self._y_test = y_test
        return X_train, X_test, y_train, y_test

    def define_model(self, **kwargs):

        model_definer = self.model_definer
        model_kwargs = utils.filter_kwargs(
            model_definer, default_kwargs=self.hyper_params, **kwargs
        )

        print(f"{model_kwargs = }")
        n_variables = len(self.features)
        n_labels = len(self.labels)
        model = model_definer(n_variables, n_labels=n_labels, **model_kwargs)
        model = self.model_modifier(model, **kwargs)
        self.model_kwargs = utils.deepcopy(model_kwargs)
        self.model = model
        return model

    def model_modifier(self, model, **kwargs):
        model_modifiers = getattr(self, "model_modifiers", [])
        for modifier in model_modifiers:
            if modifier in model_modifiers_dict:
                model = model_modifiers_dict[modifier](model, **kwargs)
            else:
                raise ValueError(
                    f"model_modifier {modifier} not recognized in {model_modifiers_dict = }"
                )
        return model

    def compile_model(
        self,
        optimizer=None,
        loss=None,
        metrics=["accuracy", "AUC"],
        optimizer_kwargs={},
        metrics_kwargs=[],
        **compile_kwargs,
    ):

        extra_optimizers = {
            "adamw": keras.optimizers.AdamW,
            "adam": keras.optimizers.Adam,
            "nadam": keras.optimizers.Nadam,
        }

        optimizer = optimizer if optimizer else self.optimizer
        optimizer = extra_optimizers.get(optimizer, optimizer)
        if isinstance(optimizer, str):
            raise ValueError("Optimizer not recognized: %s" % optimizer)
        optimizer_kwargs = utils.filter_kwargs(
            optimizer, self.hyper_params, **optimizer_kwargs
        )
        self.optimizer_kwargs = utils.deepcopy(optimizer_kwargs)

        if "learning_rate" in optimizer_kwargs:
            optimizer_kwargs["learning_rate"] = tf_utils.get_learning_rate(
                optimizer_kwargs["learning_rate"]
            )
        if "clipnorm" in optimizer_kwargs:
            optimizer_kwargs["clipnorm"] = tf_utils.get_clipnorm(
                optimizer_kwargs["clipnorm"]
            )

        loss = loss if loss else self.loss
        if loss in custom_loss_dict:
            loss_dict = custom_loss_dict[loss]
            loss = loss_dict["loss"]
            self.xy_maker = loss_dict.get("xy", None)
            if "metrics" in loss_dict:
                metrics = loss_dict["metrics"]

        # import utils.muon_embedding as muemb
        # self.xy_maker = muemb.xy_maker_muon_embedding
        # print("FIX THISSSSS")

        print(f"{optimizer_kwargs = }")
        optimizer = optimizer(**optimizer_kwargs)

        f1score = tf.keras.metrics.F1Score(average="weighted", threshold=0.5)
        # metrics = ["accu
        if metrics_kwargs:
            raise NotImplementedError()
        print(f"========== {optimizer = }")
        if hasattr(self, "weights"):
            compile_kwargs["weighted_metrics"] = metrics
        else:
            compile_kwargs["metrics"] = metrics
        print(f"{compile_kwargs = }")
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            # metrics=metrics,
            **compile_kwargs,
        )

    def get_tf_dataset(
        self, file_list, do_preproc=None, do_balance=None, balancer_kwargs={}
    ):
        """
        get the tf dataset from the given file list,
        """
        if do_preproc is None:
            do_preproc = not all(["preproc" in os.path.basename(f) for f in file_list])

        if do_balance is None:
            do_balance = not all(
                ["balanced_by" in os.path.basename(f) for f in file_list]
            )

        if do_balance:
            sample_balancer = getattr(self, "sample_balancer", None)
            if not sample_balancer:
                warnings.warn(
                    "The files don't seem to be balanced... if you need to balance them give 'sample_balancer' option."
                )
        else:
            sample_balancer = False

        print(f"DEBUG: get_tf_dataset {self.chunk_size = }")
        return tf_utils.dataset_generator(
            file_list,
            chunk_size=self.chunk_size,
            features=self.features,
            labels=self.labels,
            weights=getattr(self, "weights", None),
            preproc=do_preproc,
            balancer=sample_balancer,
            where=getattr(self, "selection", ""),
            **balancer_kwargs,
        )

    # def _dataset_generator(self, file_list, balancer=None, preproc=False, **kwargs):
    #     return tf_utils.dataset_generator(
    #         utils.get_file_list(file_list),
    #         chunk_size=self.chunk_size,
    #         features=self.features,
    #         labels=self.labels,
    #         preproc=preproc,
    #         balancer=balancer,
    #         **kwargs,
    #     )

    # def get_dataset(
    #     self,
    #     file_list=None,
    #     val_frac=0.1,
    #     test_frac=0.0,
    #     preproc=False,
    #     sample_balancer=None,
    #     balancer_kwargs={},
    #     **kwargs,
    # ):
    #     flist_train, flist_val, flist_test = utils.split_file_list(
    #         file_list, val_frac=val_frac, test_frac=test_frac
    #     )
    #     n_files = len(file_list)
    #     if not len(flist_train):
    #         raise ValueError(f"file_list_train is empty!")
    #     if test_frac and not len(flist_test):
    #         raise ValueError(
    #             f"file_list_test is empty. {test_frac=} probably too small for {n_files} files, "
    #         )
    #     if val_frac and not len(flist_val):
    #         raise ValueError(
    #             f"file_list_val is empty. {val_frac=} probably too small for {n_files} files, "
    #         )
    #     ds_train = self._dataset_generator(
    #         flist_train,
    #         preproc=preproc,
    #         balancer=sample_balancer,
    #         balancer_kwargs=balancer_kwargs,
    #         **kwargs,
    #     )
    #     ds_val = self._dataset_generator(
    #         flist_val,
    #         preproc=preproc,
    #         balancer=None,
    #         balancer_kwargs={},
    #         # balancer=sample_balancer,
    #         # balancer_kwargs=balancer_kwargs,
    #         **kwargs,
    #     )
    #     ds_test = self._dataset_generator(
    #         flist_test,
    #         preproc=preproc,
    #         balancer=None,
    #         balancer_kwargs=balancer_kwargs,
    #         **kwargs,
    #     )
    #     return dict(
    #         ds_train=ds_train,
    #         ds_val=ds_val,
    #         ds_test=ds_test,
    #         flist_train=flist_train,
    #         flist_val=flist_val,
    #         flist_test=flist_test,
    #     )

    # def get_datasets(
    #     self,
    #     file_list=None,
    #     val_frac=None,
    #     test_frac=0.0,
    #     sample_balancer=None,
    #     balancer_kwargs={},
    #     preproc=False,
    #     **kwargs,
    # ):
    #     if file_list is None:
    #         file_list = utils.get_file_list(self.fname_train)
    #     # if hasattr(self, )
    #     val_frac = val_frac if val_frac else self.validation_split
    #     di = self.get_dataset(
    #         file_list,
    #         val_frac=val_frac,
    #         test_frac=test_frac,
    #         sample_balancer=sample_balancer,
    #         balancer_kwargs=balancer_kwargs,
    #         preproc=preproc,
    #         **kwargs,
    #     )
    #     self.ds_train = di["ds_train"]
    #     self.ds_val = di["ds_val"]
    #     self.file_list_train = di["flist_train"]
    #     self.file_list_val = di["flist_val"]

    #     assert set(self.file_list_train).isdisjoint(self.file_list_val)

    #     if test_frac and di["flist_test"]:
    #         self.ds_test = di["ds_test"]
    #         self.file_list_test = di["flist_test"]

    #     elif hasattr(self, "fname_test"):
    #         file_list_test = utils.get_file_list(self.fname_test)
    #         assert set(file_list_test).isdisjoint(self.file_list_train)
    #         assert set(file_list_test).isdisjoint(self.file_list_val)

    #         self.ds_test = self._dataset_generator(
    #             file_list_test, balancer=None, preproc=preproc, **kwargs
    #         )  # don't balance the test
    #         self.file_list_test = file_list_test
    def get_sample_weight(self, df, **kwargs):
        sample_weight = None
        weight_column = getattr(self, "weights", None)
        if weight_column:
            sample_weight = df[weight_column]
        if not sample_weight is None:
            sample_weight = sample_weight.astype("float64")
        return sample_weight

    def fit(self, model=None, xy=None, **kwargs):

        is_dataset = False
        sample_weight = None
        if xy is None:
            if hasattr(self, "ds_train"):
                xy = dict(x=self.ds_train, validation_data=self.ds_val)
                is_dataset = True
                # is_tf_dataset = isinstance(xy, tf.data.Dataset)
            elif getattr(self, "xy_maker", None):
                if False:
                    if (
                        hasattr(self, "validation_split")
                        and not "validation_data" in xy
                    ):
                        xy["validation_split"] = self.validation_split
                # if hasattr(self, "validation_split") and not "validation_data" in xy:
                if hasattr(self, "validation_split"):
                    df_train, df_val = self.split_df(
                        self.df_train, frac=self.validation_split
                    )
                xy = self.xy_maker(self, df_train)
                xy_val = self.xy_maker(self, df_val)
                # xy['sample_weight'] =
                sample_weight = self.get_sample_weight(df_train)
                xy["validation_data"] = (
                    xy_val["x"],
                    xy_val["y"],
                    self.get_sample_weight(df_val),
                )
                print("DEBUG: xy_maker was called!")
            else:
                xy = dict(
                    x=self.X_train,
                    y=self.y_train,
                    validation_split=getattr(self, "validation_split", 0.2),
                )

        elif isinstance(xy, (tuple, list)):
            if len(xy) == 2:
                print("DEBUG xy is a tuple of length 2")
                xy = dict(x=xy[0], y=xy[1])
            elif len(xy) > 2:
                print("DEBUG xy is a tuple of length 2")
                xy = dict(x=xy[0], y=xy[1:])
        elif isinstance(xy, dict):
            pass
        elif callable(xy):
            print("DEBUG: xy was given as function, will call it like `xy(self)` ")
            xy = xy(self)
        else:
            raise ValueError(
                f"xy should be a tuple or a list of length 2 or more, but got {xy}"
            )

        model = model if model else self.model
        if not is_dataset and sample_weight is None:
            weight_column = getattr(self, "weights", None)
            if "sample_weight" in kwargs:
                sample_weight = kwargs.pop("sample_weight")
                raise NotImplementedError(
                    "sample_weight in kwargs is not implemented yet"
                )
            elif weight_column:
                sample_weight = self.df_train[weight_column]
        if not sample_weight is None:
            sample_weight = sample_weight.astype("float64")
        fit_kwargs = dict(
            epochs=kwargs.pop("n_epoch", self.n_epoch),
            batch_size=kwargs.pop("batch_size", self.batch_size),
            class_weight=kwargs.pop("class_weight", self.class_weight),
            sample_weight=sample_weight,
        )
        fit_kwargs.update(**kwargs)
        fit_kwargs["verbose"] = 1 if "cobalt" in utils.get_hostname() else 2

        debug_weights = getattr(self, "debug_weights", False)
        if debug_weights:
            fit_kwargs["callbacks"] = [
                tf_utils.make_epoch_checkpoint_callback(self.model_dir)
            ]

        print(f"{fit_kwargs = }")
        # print(f"{xy = }")
        print("--------------------")
        print(xy.keys())
        print("--------------------")
        print(xy)
        history = model.fit(
            **xy,
            **fit_kwargs,
            # tf_utils.epoch_checkpoint_callback,
        )
        self.history = history
        return history

    @property
    def model_optimizer_kwargs(self):

        if not hasattr(self, "model_kwargs"):
            self.define_model()
        model_kwargs = self.model_kwargs

        if not hasattr(self, "optimizer_kwargs"):
            self.compile_model()
        optimizer_kwargs = self.optimizer_kwargs

        return list(model_kwargs) + list(optimizer_kwargs)

    """
        Hyper parameter tunning
    """

    def hyper_tune(
        self,
        n_epochs=[20, 50],
        methods=["hyperband", "bayesian", "randomsearch"],
        verbose=True,
        **kwargs,
    ):
        commands_all = []
        res = {}

        for method in methods:
            for n_epoch in n_epochs:
                tuner, commands = self._hyper_tune(
                    tuner=method, n_epoch=n_epoch, **kwargs
                )
                commands_all.append(
                    dict(method=method, n_epoch=n_epoch, commands=commands)
                )
                res[method] = {"tuner": tuner, "commands": commands}

        print("\n\n### Commands: ")
        for di in commands_all:
            print(f"\n#  {di['method'] = } {di['n_epoch'] = }")
            print("\n".join(di["commands"]))
        return res

    def _hyper_tune(
        self,
        tuner="hyperband",
        objective="val_loss",
        max_trials=50,
        hp_n_epoch=50,
        n_epoch=None,
        model_modifier=None,
        metrics=["accuracy"],
        overwrite=False,
        model_builder_wrapper=None,
        search_callbacks=[],
    ):
        import utils.hyper_tune
        from utils.hyper_tune import (
            kt,
            TimeStopping,
            ModelSizeMonitor,
            get_cmd_arg,
        )

        model_builder_wrapper = (
            model_builder_wrapper
            if model_builder_wrapper
            else utils.hyper_tune.model_builder_wrapper
        )

        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor=objective,  # Monitor validation loss
            min_delta=1e-6,  # Minimum change to qualify as an improvement
            patience=2,  # How many epochs to wait for improvement
            mode="auto",  # Direction of improvement is auto-inferred
            restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
            verbose=1,
        )

        time_stopping_callback = TimeStopping(max_seconds=300, verbose=1)
        model_size_callback = ModelSizeMonitor(
            min_layers=5, max_layers=50, max_params=1_000_000
        )

        model_builder = model_builder_wrapper(
            self, model_modifier=model_modifier, metrics=metrics
        )
        if tuner.lower() == "hyperband":
            tuner_label = "hyperband"
            tuner = kt.Hyperband(
                model_builder,
                objective=objective,
                max_epochs=20,
                factor=3,
                directory=os.path.join(self.model_dir, "keras_tuner"),
                project_name="hyperband",
                overwrite=overwrite,
                # overwrite=args.overwrite
                # batch_size=1000,
            )

        elif tuner.lower() in ["bayesian", "bayesianoptimizer"]:
            tuner_label = "bayesian"
            tuner = kt.BayesianOptimization(
                model_builder,
                objective=objective,
                # max_epochs=20,
                # factor=3,
                directory=os.path.join(self.model_dir, "keras_tuner"),
                project_name="bayesianoptimization",
                max_trials=max_trials,
                num_initial_points=None,
                alpha=0.0001,
                beta=2.6,
                seed=None,
                overwrite=overwrite,
                # overwrite=args.overwrite,
                # hyperparameters=None,
                # tune_new_entries=True,
                # allow_new_entries=True,
                # max_retries_per_trial=0,
                # max_consecutive_failed_trials=3,
                # **kwargs
            )

        elif tuner.lower() == "randomsearch":
            tuner_label = "randomsearch"
            tuner = kt.RandomSearch(
                model_builder,
                objective=objective,
                max_trials=max_trials,
                directory=os.path.join(self.model_dir, "keras_tuner"),
                project_name="randomsearch",
                overwrite=overwrite,
                # overwrite=args.overwrite
                # **kwargs
            )
        else:
            raise ValueError(f"requested tuner `{tuner}` not recognized.")

        if hasattr(self, "xy_maker"):
            xy = self.xy_maker(self)
        else:
            xy = dict(x=self.X_train, y=self.y_train)

        tuner.search(
            **xy,
            # self.X_train,
            # self.y_train,
            epochs=hp_n_epoch if hp_n_epoch else self.n_epoch,
            batch_size=self.batch_size,
            callbacks=[
                model_size_callback,
                early_stopping_callback,
                time_stopping_callback,
            ]
            + search_callbacks,
            sample_weight=(
                None
                if not getattr(self, "weights", None)
                else self.df_train[self.weights]
            ),
            validation_split=getattr(self, "validation_split", 0.2),
        )
        best_hp = tuner.get_best_hyperparameters()[0]
        print("best params:", best_hp.values)
        print("args: %s" % get_cmd_arg(best_hp))
        commands = self.make_hypertuned_config(
            tuner, n_best=5, label=tuner_label, n_epoch=n_epoch
        )
        return tuner, commands

    def test_model(self, model=None):
        model = model if model else self.model
        pred = model.predict(x=self.X_test, batch_size=self.batch_size)
        return pred
        # model.evaluate(self.X_test, self.y_test)

    def make_hypertuned_config(self, tuner, n_best=1, label="hypertuned", n_epoch=None):
        """
        update the current hyper_params based on the tuner results and create a new config file

        """
        from utils.hyper_tune import get_cmd_arg

        best_hps = tuner.get_best_hyperparameters(n_best * 2)

        config_paths = []
        hps = []
        ihp = 0
        commands = []
        for best_hp_params in best_hps:
            # best_hp = {k:v for k,v in best_hp.values.items() if not k.startswith("tuner/")}
            # best_hp = get_cmd_arg(best_hp.values, return_dict=True)
            best_hp = utils.deepcopy(self.hyper_params)
            best_hp.update(**get_cmd_arg(best_hp_params.values, return_dict=True))

            model_name = self.model_name + "_%s%s" % (label, ihp)
            if n_epoch is not None:
                best_hp["n_epoch"] = n_epoch
                model_name += f"_{n_epoch}epochs"
                extra_args = "--n_epoch %s" % n_epoch

            config_path = os.path.join(
                self.model_dir_base,
                self.version_tag,
                self.model_tag,
                model_name,
                "config.yml",
            )

            if best_hp in hps:
                continue

            hps.append(utils.deepcopy(best_hp))

            best_hp.update(
                model_dir=os.path.join(
                    self.model_dir_base, self.version_tag, self.model_tag, model_name
                ),
                plot_dir=os.path.join(
                    self.plot_dir_base, self.version_tag, self.model_tag, model_name
                ),
                model_name=model_name,
            )
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            self.save_config(
                config_path,
                clean_kwargs=True,
                **best_hp,
            )
            command = f"python ModelFactory.py --config {config_path}"
            commands.append(command)
            config_paths.append(config_path)
            ihp += 1
            if len(config_paths) >= n_best:
                break

        # for hp in hps:
        #     print(hp)
        # print("\n".join(commands))
        return commands

    """
        Validation methods
    """

    def get_sample_idx(self, N=10000, reuse=False):
        if (
            reuse
            and hasattr(self, "_train_sample_idx")
            and hasattr(self, "_test_sample_idx")
        ):
            return self._train_sample_idx, self._test_sample_idx
        N_train = N if N < len(self.df_train) else len(self.df_train)
        train_sample_idx = self.df_train.sample(N_train).index
        N_test = N if N < len(self.df_test) else len(self.df_test)
        test_sample_idx = self.df_test.sample(N_test).index
        print(f"{N_test = } , {N_train = }")
        self._train_sample_idx = train_sample_idx
        return train_sample_idx, test_sample_idx

    def get_pred(
        self,
        column="pred",
        include_train=True,
        nsample=0,
        pred_func=None,
        reuse=False,
    ):
        pred_args = {}
        pred_func = pred_func if pred_func else self.model.predict
        if getattr(self, "batch_size"):
            pred_args["batch_size"] = self.batch_size

        # if reuse:
        #     if hasattr(self, "df_train_sample") and hasattr(self, "df_test_sample"):
        #         df_train = self.df_train_sample
        #         df_test = self.df_test_sample
        #     else:
        #         self.get_pred(column=column, include_train=include_train, nsample=nsample, pred_func=pred_func, reuse=False)

        if nsample:
            ## FIXME this fails if X_train needs a xy_maker!
            train_idx, test_idx = self.get_sample_idx(nsample, reuse=reuse)
            if (
                reuse
                and hasattr(self, "df_train_sample")
                and hasattr(self, "df_test_sample")
            ):
                df_train = self.df_train_sample
                df_test = self.df_test_sample
            else:
                df_train = self.df_train.loc[train_idx]
                df_test = self.df_test.loc[test_idx]

            df_train[column] = pred_func(
                self.X_train.loc[train_idx],
                **pred_args,
            )
            df_train["truth"] = self.y_train.loc[train_idx]
            df_test[column] = pred_func(
                self.X_test.loc[test_idx],
                **pred_args,
            )
            df_test["truth"] = self.y_test.loc[test_idx]

            self.df_train_sample = df_train
            self.df_test_sample = df_test
            # return df_train, df_test

        else:
            df_train = self.df_train
            df_test = self.df_test
            if include_train:
                df_train[column] = pred_func(self.X_train, **pred_args)
                df_train["truth"] = self.y_train
            df_test[column] = pred_func(self.X_test, **pred_args)
            df_test["truth"] = self.y_test
            self.df_train_sample = df_train
            self.df_test_sample = df_test

    def plot_predictions(
        self,
        # df1, df2
        variable="pred",
        plot_name=None,
        bins=50,
        range=(0, 1),
        plot_path=None,
        query=None,
        nsample=100_000,
        weights=None,
        **kwargs,
    ):
        if variable == "pred":
            plot_name = plot_name if plot_name else "predictions"
            x_label = "Predictions"
        else:
            plot_name = plot_name if plot_name else variable
            x_label = variable

        plot_path = (
            plot_path
            if plot_path != None
            else os.path.join(self.plot_dir, "%s.png" % plot_name)
        )
        kwargs.setdefault("x_label", x_label)
        kwargs.setdefault("y_scale", "log")
        # kwargs.setdefault("")

        if nsample:
            # df_train, df_test =
            if not hasattr(self, "df_test_sample"):
                self.get_pred(nsample=nsample)
            df_test = self.df_test_sample.query(query) if query else self.df_test_sample
            df_train = (
                self.df_train_sample.query(query) if query else self.df_train_sample
            )

        else:
            df_test = self.df_test.query(query) if query else self.df_test
            df_train = self.df_train.query(query) if query else self.df_train
        return utils.compare_test_train(
            df_test,
            df_train,
            variable=variable,
            bins=bins,
            range=range,
            plot_path=plot_path,
            y_label_ratio="pull",
            weight_var=getattr(self, "weights", None) if weights is None else weights,
            **kwargs,
        )

    def pairplot(
        self,
        df=None,
        label=None,
        N=5000,
        features=None,
        plot_kws={"alpha": 0.1, "s": 5, "edgecolor": "k"},
        plot_name="pairplot_{label}.png",
        **kwargs,
    ):
        import seaborn as sns

        df = df if not df is None else self.df_train.sample(N)
        label = label if label else self.labels[0]
        features = features if features else self.features
        kwargs.setdefault("corner", True)

        n_cat = len(df[label].unique())
        palette = sns.color_palette("coolwarm_r", n_cat)

        pairplot = sns.pairplot(
            df, hue=label, vars=features, plot_kws=plot_kws, palette=palette, **kwargs
        )

        handles = pairplot._legend_data.values()
        labels = pairplot._legend_data.keys()

        for lh in handles:
            lh.set_alpha(1)
            lh.set_markersize(10)

        pairplot._legend.remove()

        fig = plt.gcf()
        fig.legend(
            handles,
            labels,
            title="prediction quantiles",
            loc="upper center",
            ncol=n_cat,
            bbox_to_anchor=(0.5, 0.95),
            fontsize=18 + len(features),
        )

        plot_name = plot_name.format(label=label)
        utils.savefig(fig, os.path.join(self.plot_dir, plot_name))

    def pairplot_prediction_categories(
        self,
        use_train=True,
        features=None,
        **kwargs,
    ):
        features = features if features else self.features

        attr_name = "df_train_sample" if use_train else "df_test_sample"
        if not hasattr(self, attr_name):
            self.get_pred()

        df = getattr(self, attr_name)

        n_cats = 4
        cats, edges = pd.qcut(
            df["pred"], n_cats, labels=["1", "2", "3", "4"], retbins=True
        )

        hue = "pred_cats"
        df[hue] = cats
        print(edges)
        self.pairplot(df, label=hue, features=features, **kwargs)

    def get_shap_values(self, nsample=100, X=None, features=None):
        """ """
        import shap

        X = X if X else self.X_train  # or X_test?!
        X = X if not nsample else X.sample(nsample)
        features = features if features else self.features

        explainer = shap.Explainer(self.model.predict, X)
        shap_values = explainer(X)
        self.shap_values = shap_values
        return shap_values

    def get_shap_feature_importance(self, shap_values=None, list_only=False):
        if not shap_values:
            if hasattr(self, "shap_values"):
                shap_values = self.shap_values
            else:
                shap_values = self.get_shap_values()

        feature_names = self.features
        shap_df = pd.DataFrame(shap_values.values, columns=feature_names)

        means = np.abs(shap_df.values).mean(0)

        shap_importance = pd.DataFrame(
            list(zip(feature_names, means)), columns=["feature", "shap_importance"]
        )
        shap_importance.sort_values(
            by=["shap_importance"], ascending=False, inplace=True
        )

        shap_importance = shap_importance.reset_index(drop=True)
        if list_only:
            return shap_importance.feature.to_list()
        else:
            return shap_importance

    def make_shap_plots(
        self,
        shap_values=None,
        nsample=100,
        X=None,
        features=None,
        max_display=None,
        name_tag="",
    ):
        """ """
        import shap

        if not shap_values:
            shap_values = getattr(self, "shap_values", None)
            if not shap_values:
                self.get_shap_values(nsample=nsample, X=X, features=features)

            # shap_values = getattr(
            #     self,
            #     "shap_values",
            #     self.get_shap_values(nsample=nsample, X=X, features=features),
            # )

        shap_plot_dir = self.plot_dir
        # summary plot
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, max_display=max_display, show=False)
        utils.savefig(
            plt,
            os.path.join(
                shap_plot_dir,
                "shap_summary%s.png" % ("_%s" % name_tag if name_tag else ""),
            ),
        )

        # beeswarm plot
        fig, ax = plt.subplots()
        shap.plots.beeswarm(
            shap_values,
            max_display=max_display,
            alpha=0.75,
            show=False,
        )
        utils.savefig(
            plt,
            os.path.join(
                shap_plot_dir,
                "shap_beeswarm%s.png" % ("_%s" % name_tag if name_tag else ""),
            ),
        )

    def feature_importance(self, nsample=10_000, n_repeats=10):
        X = self.X_train.sample(nsample)
        y = self.y_train.sample(nsample)
        features = self.features

        imp_di = tf_utils.get_feature_permutation_importance(
            self.model, X, y, features=features, N=None, n_repeats=10
        )

        plot_dir = self.plot_dir
        for k, v in imp_di.items():
            fig = v["fig"]
            utils.savefig(fig, os.path.join(plot_dir, "feature_importance_%s.png" % k))
        return imp_di

    def plot_history(self, history=None, **kwargs):
        history = history if history else self.history
        fig, ax, df_hist = tf_utils.plot_history(history, **kwargs)
        utils.savefig(fig, os.path.join(self.plot_dir, "history.png"))

        history = getattr(history, "history", history)
        metrics = list(history.keys())
        metric_pairs = {
            k: (
                {k: history[k], "val_%s" % k: history["val_%s" % k]}
                if "val_%s" % k in metrics
                else {k: history[k]}
            )
            for k in metrics
            if not k.startswith("val_")
        }

        for icolor, (label, hist) in enumerate(metric_pairs.items()):
            fig, ax, df_hist = tf_utils.plot_history(
                hist, ylim=None, plot_kwargs=dict(color="C%s" % icolor), **kwargs
            )
            ax.set_ylabel(label)
            vals = df_hist.values.flatten()
            ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.01)
            # ax.set_ylim(min(vals), max(vals))
            utils.savefig(fig, os.path.join(self.plot_dir, "history_%s.png" % label))
        return df_hist

    def plot_roc(self):
        fig, ax = plt.subplots()
        if not hasattr(self, "df_test_sample"):
            self.get_pred()

        weight_column = getattr(self, "weights", None)
        df = self.df_train_sample
        roc_train = utils.plot_roc(
            (fig, ax),
            truth=df["truth"],
            pred=df["pred"],
            weights=df[weight_column] if weight_column else None,
            label="train",
        )
        # roc_train['line'].set_la
        df = self.df_test_sample
        roc_test = utils.plot_roc(
            (fig, ax),
            truth=df["truth"],
            pred=df["pred"],
            weights=df[weight_column] if weight_column else None,
            label="test",
        )
        ax.set_ylim(ax.get_ylim()[0], 1.2)
        ax.legend(loc="upper left", ncol=2)
        utils.savefig(fig, os.path.join(self.plot_dir, "ROC.png"))
        return fig, ax, roc_train, roc_test

    def model_summary(self, model=None, html=False, file_path=None):
        model = model if not model is None else self.model
        model_text = tf_utils.get_model_summary(model)
        if html:
            html_template = """<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Model Summary</title>
                <style>
                    body {
                        background-color: #121212; /* Dark background */
                        color: #ffffff;           /* White text */
                        font-family: "Courier New", Courier, monospace; /* Monospace font for better alignment */
                        padding: 20px;           /* Add some spacing around the content */
                    }
                    pre {
                        white-space: pre-wrap;   /* Preserve formatting and wrap long lines */
                        word-wrap: break-word;  /* Break long words to fit the screen */
                    }
                </style>
            </head>
            <body>
            <pre>
            %s
            </pre>
            </body>
            </html>"""
            # model_text = tf_utils.get_model_summary(self.model)
            model_text = html_template % model_text

        if file_path:
            with open(file_path, "w") as f:
                print(model_text, file=f)
        else:
            return model_text

    def plot_prc(self):
        fig, ax = plt.subplots()
        if not hasattr(self, "df_test_sample"):
            self.get_pred()

        weight_column = getattr(self, "weights", None)
        df = self.df_train_sample
        pr_train = utils.plot_prc(
            (fig, ax),
            truth=df["truth"],
            pred=df["pred"],
            weights=df[weight_column] if weight_column else None,
            label="train",
        )
        # roc_train['line'].set_la
        df = self.df_test_sample
        pr_test = utils.plot_prc(
            (fig, ax),
            truth=df["truth"],
            pred=df["pred"],
            weights=df[weight_column] if weight_column else None,
            label="test",
        )
        ax.set_ylim(ax.get_ylim()[0], 1.2)
        ax.legend(loc="upper right", ncol=2)
        utils.savefig(fig, os.path.join(self.plot_dir, "precision_recall.png"))
        return fig, ax, pr_train, pr_test

    def plot_confusion(self, probs=[0.4, 0.6, 0.7, 0.8]):
        """
        plot confusion matrix and the given probs
        """
        n_probs = len(probs)

        if not hasattr(self, "df_test_sample"):
            self.get_pred()
        for name, df in [
            ("test", self.df_test_sample),
            ("train", self.df_train_sample),
        ]:
            fig, axs = plt.subplots(
                ncols=2,
                nrows=int(n_probs / 2),
                figsize=(12, 5 * n_probs / 2),
                squeeze=False,
            )
            fig.tight_layout()
            plt.subplots_adjust(hspace=0.3)

            y = df["truth"]
            pred = df["pred"]

            for prob, (ax) in zip(probs, axs.flatten()):
                tf_utils.plot_confusion(
                    y,
                    pred,
                    prob=prob,
                    normalize="true",
                    ax=ax,
                    title=f"Confusion Matrix ({name}):\n p>{prob}",
                    sample_weight=(
                        None if not getattr(self, "weights") else df[self.weights]
                    ),
                    verbose=False,
                )

            utils.savefig(fig, os.path.join(self.plot_dir, "confusion_%s.png" % name))

    def plot_speed_up(
        self,
        nom="n_eff_passed",
        denom="n_photons_simulated",
        plot_name="speed_up",
        col_gen_weights="flux_weights",
        col_sel_weights="selection_weights",
        pred_name="pred",
        save_res=None,
        df=None,
    ):

        import utils.speedup_utils as speedup

        if df is None:
            if not hasattr(self, "df_test_sample"):
                self.get_pred(pred_name=pred_name)
            df = self.df_test_sample

        speedup.add_pseudo_preds(df)
        df["passed"] = df["truth"] == 1
        res = {
            k: speedup.get_pred_speed_up(
                df,
                pred=k,
                col_gen_weights=col_gen_weights,
                col_sel_weights=col_sel_weights,
            )
            for k in ["pseudo_pred_gauss", "pseudo_pred_uniform", pred_name]
        }
        fig, ax = speedup.make_tri_plot(
            nom=nom, denom=denom, sdfs=res, label_name=self.labels[0]
        )
        if plot_name:
            utils.savefig(fig, f"{self.plot_dir}/{plot_name}.png")
        if save_res:
            save_res = (
                save_res
                if isinstance(save_res, str)
                else os.path.join(self.model_dir, "speedup.pkl")
            )
            pickle.dump(res, open(save_res, "wb"))
            # utils.savefig(fig, save_path
        return res

    def plot_comp_cost(
        self,
        df=None,
        pred_name="pred",
        sim_levels=["generated", "triggered", "filtered", "Level3"],
        key="sim_level",
        plot_path=None,
        plot_kwargs={},
    ):
        """
        make scatter plot of the weighted computation cost vs the prediction score for different simulation levels
        """
        if df is None:
            df = self.df_test_sample

        fig, ax = plt.subplots()
        colors = {
            "generated": "C1",
            "triggered": "C2",
            "filtered": "C5",
            "Level3": "C0",
        }
        if key not in df.columns:
            utils.label_sim_levels(df, sim_levels=sim_levels, key=key)

        kwargs = dict(s=0.2, alpha=0.1)
        kwargs.update(plot_kwargs)

        for level in sim_levels:
            df_ = df.query(f'{key}=="{level}"')
            vals = df_[pred_name] * df_["selection_weights"] * df_["n_photons"]
            ax.scatter(
                df_[pred_name],
                vals,
                label=level,
                color=colors.get(level, None),
                **kwargs,
            )

            ax.set_yscale("log")
            ax.set_xlabel("sampling probability ($p_{i}$)")
            ax.set_ylabel(
                "weighted comp. cost \n($w^{sel.}_{i} \cdot p_{i} \cdot n_{\gamma,\ i}$)",
                fontsize=16,
            )

        leg = ax.legend(ncols=2, loc="upper left")
        for handle in leg.legend_handles:
            handle.set_alpha(1)
            handle.set_sizes([20])
        if plot_path:
            utils.savefig(fig, plot_path, formats=["png"])
        return fig, ax

    def plot_cumsum_frac(self, bins=50, range=(0, 1), density=True, col_name="pred"):
        label = self.labels[0]
        pos_cond = f"{label}==1"
        neg_cond = f"{label}==0"

        if not hasattr(self, "df_test_sample"):
            self.get_pred()

        df_test = self.df_test_sample
        df_train = self.df_train_sample

        ret = {}
        for samp_name, df in (("test", df_test), ("train", df_train)):
            for cond_name, cond in (("positive", pos_cond), ("negative", neg_cond)):
                name = f"{cond_name}_{samp_name}"
                hcs = utils.get_cumsum_frac(
                    df.query(cond)[col_name], bins=bins, range=range, density=density
                )
                ret[name] = hcs
                # hcs.label = name

        fig, ax = plt.subplots()

        areas = {}
        for name, h in ret.items():
            centers = h.axes[0].centers
            label = name.split("_")
            label = f"{label[0]} ({label[1]})"
            ax.plot(
                centers,
                h.counts(),
                ls=":" if "train" in name else "-",
                color="C0" if "positive" in name else "C1",
                label=label,
            )
            areas[name] = round(np.trapz(h, h.axes[0].centers), 3)

        ax.set_xlabel("selection threshold")
        ax.set_ylabel("cumilative fraction")
        ax.set_ylim(-0.1, 1.4)
        xwidth = abs(range[1] - range[0])
        ax.set_xlim(range[0] - xwidth * 0.1, range[1] + xwidth * 0.1)
        ax.legend(loc="upper left", ncols=2)

        area_diffs = {}
        for samp_name in ["train", "test"]:
            area_diffs[samp_name] = round(
                areas["negative_%s" % samp_name] - areas["positive_%s" % samp_name], 3
            )
        print(area_diffs)

        cfd_str = str("test:  %0.3f" % area_diffs["test"])
        cfd_str += str("\ntrain: %0.3f" % area_diffs["train"])

        ax.text(
            -0.05,
            1.08,
            "area differences:\n%s" % cfd_str,
            verticalalignment="top",
            fontsize=11,
        )
        utils.savefig(fig, os.path.join(self.plot_dir, "cumilative_fraction.png"))
        return fig, ax, ret

    def generate_html(self):
        utils.generate_html_from_dir(self.plot_dir)


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    def non_or_bool_type(value):
        if value.lower() == "none":
            return None
        elif value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        return value

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--fnames",
        nargs="+",
        default=[default_config["fname_train"]],
        help="input file pattern ",
    )
    parser.add_argument(
        "--fname_train",
        nargs="+",
        default=[default_config["fname_train"]],
        help="input file for the training",
    )
    parser.add_argument(
        "--fname_test",
        nargs="+",
        default=[default_config["fname_test"]],
        help="input file for the testing",
    )
    # parser.add_argument(
    #     "--fname_val", default=None, help="input file for the validation"
    # )

    parser.add_argument("--version_tag", default=default_config["version_tag"])
    parser.add_argument("--model_tag", default=default_config["model_tag"])
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--n_epoch", default=default_config["n_epoch"], type=int)
    parser.add_argument("--init_layer", default=default_config["init_layer"], type=int)
    parser.add_argument(
        "--final_layer", default=default_config["final_layer"], type=int
    )
    parser.add_argument("--max_layers", default=30, type=int)
    parser.add_argument(
        "--layer_repeat", default=default_config["layer_repeat"], type=int
    )
    parser.add_argument("--unit_step", default=default_config["unit_step"], type=int)
    parser.add_argument("--seed", default=default_config["seed"], type=int)
    parser.add_argument("--batch_size", default=default_config["batch_size"], type=int)
    parser.add_argument("--optimizer", default=default_config["optimizer"])
    parser.add_argument(
        "--learning_rate", default=default_config["learning_rate"], type=str
    )
    parser.add_argument("--dropout", default=default_config["dropout"], type=float)
    parser.add_argument("--hidden_activation", default="relu")
    parser.add_argument(
        "--features",
        default=DEFAULT_FEATURES,
        nargs="+",
        help="name of the feature or a feature set",
    )
    parser.add_argument("--add_features", default=[], nargs="+")
    parser.add_argument("--remove_features", default=[], nargs="+")
    # parser.add_argument("--feature_sets", default=default_config['feature_sets'], nargs="+")
    parser.add_argument("--labels", default=default_config["labels"], nargs="+")
    parser.add_argument("--weight_decay", type=float)
    # parser.add_argument(
    #     "--ema_momentum", type=float, default=default_config["ema_momentum"]
    # )
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--weights", type=non_or_bool_type, default=None)
    parser.add_argument("--selection")
    parser.add_argument("--validate", action="store_true")
    # parser.add_argument('--n_epoch', default=10, type=int)

    # parser.add_argument("--presel_train", default="triggered")
    parser.add_argument("--basic_plots_only", action="store_true")
    parser.add_argument("--presel_train", default="")
    parser.add_argument("--sample_balancer", default="")
    parser.add_argument("--balanced_by", default="filtered")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clipnorm", type=float, default=1.0)
    parser.add_argument("--do_hyper_tune", action="store_true")
    parser.add_argument(
        "--config",
        help="path to the config.yml file containing arguments. all other args will be ignored",
    )

    parser.add_argument(
        "--model_modifiers",
        default=[],
        nargs="+",
        help="model modifier functions",
        choices=list(model_modifiers_dict),
    )

    parser.add_argument("--loss", default=default_config["loss"], type=str)

    args = parser.parse_args()

    tf_utils.set_seeds(args.seed)

    features = select_features(
        args.features,
        add_features=args.add_features,
        remove_features=args.remove_features,
    )

    train_input_tags = []

    if args.presel_train:
        train_input_tags.append("presel_%s" % args.presel_train)
    if args.balanced_by:
        train_input_tags.append("balanced_by_%s" % args.balanced_by)

    train_input_tag = "_".join(train_input_tags)
    train_input_tag = "_%s" % train_input_tag if train_input_tag else ""

    fname_train = [
        f.format(
            version_tag=args.version_tag,
            train_input_tag=train_input_tag,
        )
        for f in args.fname_train
    ]

    fname_test = [f.format(version_tag=args.version_tag) for f in args.fname_test]

    # fname_val = (
    #     None
    #     if not args.fname_val
    #     else args.fname_val.format(
    #         version_tag=args.version_tag,
    #         train_input_tag=train_input_tag,
    #     )
    # )

    if not args.config:
        mf = ModelFactory(
            version_tag=args.version_tag,
            model_tag=args.model_tag,
            fname_train=fname_train,
            fname_test=fname_test,
            # fname_val=fname_val,
            features=features,
            model_name=args.model_name,
            labels=args.labels,
            weights=args.weights,
            init_layer=args.init_layer,
            final_layer=args.final_layer,
            max_layers=args.max_layers,
            layer_repeat=args.layer_repeat,
            unit_step=args.unit_step,
            batch_size=args.batch_size,
            n_epoch=args.n_epoch,
            optimizer=args.optimizer,
            # learning_rate=tf_utils.lr_dict.get(args.learning_rate, float(args.learning_rate)),
            # learning_rate=tf_utils.lr_dict.get(args.learning_rate),
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            hidden_activation=args.hidden_activation,
            # ema_momentum=args.ema_momentum,
            weight_decay=args.weight_decay,
            selection=args.selection,
            seed=args.seed,
            clipnorm=args.clipnorm,
            sample_balancer=args.sample_balancer,
            loss=args.loss,
        )
    else:
        ## need to deal with the modified args, like fname_test, etc
        print(f"\nLoading config file: {args.config}\n")
        specified_args = utils.get_specified_args(args, parser)
        print(f"========= {specified_args = } =========")
        mf = ModelFactory.load_config(args.config, **specified_args)
        # assert False
        # if not os.path.isfile(args.config):
        #     raise ValueError(f"Config file could not be found {args.config=}")
        # import yaml

        # config_file = yaml.safe_load(open(args.config, "r"))
        # mf = ModelFactory(**config_file)

    if not os.path.isfile(os.path.join(mf.model_dir, "model.keras")) or args.overwrite:
        mf.define_model()
        mf.compile_model()
        print(tf_utils.get_model_summary(mf.model))

        if not args.interactive:
            mf.fit()
            mf.plot_history()
            mf.save_model()
            # mf.plot_predictions(plot_name="predictions_reweighted_to_test")
            mf.plot_predictions(reweight_train_to_test=False, plot_name="predictions")
            # for weight_var in ['gen_weights_balanced', 'selection_weight', 'sel_gen_weights']:

            weight_vars = ["sel_flux_weights"] + ([mf.weights] if mf.weights else [])
            # if not args.basic_plots_only:
            #     weight_vars += ['selection_weight', 'weights'] +
            for weight_var in weight_vars:
                if weight_var in mf.df_test.columns:
                    mf.plot_predictions(
                        weights=weight_var,
                        plot_name=f"predictions_{weight_var}",
                        reweight_train_to_test=False,
                    )
            if not args.basic_plots_only:
                mf.plot_confusion()
                mf.plot_roc()
                mf.plot_prc()
                mf.plot_cumsum_frac()
                mf.plot_speed_up(
                    nom="n_eff_passed",
                    denom="n_photons_simulated",
                    plot_name="speed_up",
                )
                mf.plot_speed_up(
                    nom="n_eff_noflux_passed",
                    denom="n_photons_simulated",
                    plot_name="speed_up_nofluxweight",
                )
                mf.plot_comp_cost(
                    df=mf.df_test_sample,
                    plot_path=os.path.join(mf.plot_dir, "comp_cost.png"),
                    plot_kwargs={"s": 0.2, "alpha": 0.1},
                )
                mf.generate_html()
            if args.do_hyper_tune:
                mf.hyper_tune()

    else:

        print("Loading model from %s" % mf.model_dir)
        mf = ModelFactory.load_model(
            mf.model_name,
            model_dir_base=mf.model_dir_base,
            model_tag=mf.model_tag,
            version_tag=mf.version_tag,
        )
        print("Model already exists")
