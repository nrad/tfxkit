import logging
import pandas as pd
from tfxkit.common.base_utils import read_hdfs
import tfxkit.common.tf_utils as tf_utils

logger = logging.getLogger(__name__)

class DataManager:
    """
    Responsible for loading data files, preparing training and testing sets,
    and constructing tf.data.Dataset pipelines.
    """

    def __init__(self, config):
        self.config = config
        self.train_df = None
        self.test_df = None
        self.__add_cached_df()

    def load_df(self, files=[]):
        """Load raw data into pandas DataFrames from HDF5 files."""
        logger.info(f"Loading data from files: {files}")
        df = read_hdfs(
            files,
            postselection=self.config.get("selection", None),
        )
        return df

    @classmethod
    def __add_cached_df(cls):
        """
        dynamically add properties by caching the values
        using the appropriate functions
        for example this adds a property df_train which just gets attribute _df_train if it exists
        or calls load_df and sets it if it doesn't exist.
        """

        for attr_name in ["X_test", "X_train", "y_test", "y_train"]:
            def func(self, attr_name=attr_name):
                if not hasattr(self, f"_{attr_name}"):
                    self.prep_Xy()
                return getattr(self, f"_{attr_name}")
            setattr(cls, attr_name, property(func))

        for test_train in ["test", "train"]:

            df_attr_name = f"df_{test_train}"
            logger.debug(f"Accessing {df_attr_name} with files_key: {test_train}_files")


            def func(self, test_train=test_train):
                files_key = f"{test_train}_files"
                files = self.config[files_key]
                if not hasattr(self, f"_{df_attr_name}"):
                    attr = self.load_df(files)
                    # assert False, f"{df_attr_name} is not set correctly {test_train}"
                    setattr(self, f"_{df_attr_name}", attr)
                return getattr(self, f"_{df_attr_name}")

            # assert False, f"{df_attr_name} is not set correctly"
            # print(f"DEBUG: Adding property {df_attr_name} to DataManager")
            logger.debug(f"Adding property {df_attr_name} to DataManager")
            setattr(cls, df_attr_name, property(func))

    def prep_Xy(self):

        if hasattr(self, "_X_train"):
            return self._X_train, self._X_test, self._y_train, self._y_test

        elif hasattr(self, "xy_maker"):
            Xy_train = self.xy_maker(self, self.df_train, self.config)
            Xy_test = self.xy_maker(self, self.df_test, self.config)
            X_train, X_test, y_train, y_test = (
                Xy_train["x"],
                Xy_test["x"],
                Xy_train["y"],
                Xy_test["y"],
            )

        else:
            X_train, y_train = tf_utils.xy_maker(self.df_train, self.config.features, self.config.labels)
            X_test, y_test = tf_utils.xy_maker(self.df_test, self.config.features, self.config.labels)

        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test

        return X_train, X_test, y_train, y_test

    def prepare_datasets(self):
        """Convert raw data into tf.data.Dataset objects."""
        pass
