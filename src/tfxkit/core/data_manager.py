import logging
import pandas as pd
from tfxkit.common.base_utils import read_hdfs, import_function
from dataclasses import dataclass, field
import tfxkit.common.tf_utils as tf_utils
from importlib import resources
import os
import glob


logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit:
    df: object = None
    X: object = None
    y: object = None
    sample_weight: object = None


class DataManager:
    """
    Responsible for loading data files, preparing training and testing sets,
    and constructing tf.data.Dataset pipelines.
    """

    def __init__(self, config):
        self.config = config
        self.data_config = self.config.data
        # self.sample_weight_column = self.data_config.get("sample_weight", None)
        self.__add_cached_df()

    def get_file_reader(self):
        """Dynamically import and return the file reader function."""
        file_reader = import_function(self.data_config.file_reader)
        return file_reader

    def _load_df(self, files=[]):
        """Load raw data into pandas DataFrames from HDF5 files."""

        files = [files] if isinstance(files, str) else files

        file_reader = self.get_file_reader()
        df = file_reader(files)
        # df = read_hdfs(
        #     files,
        #     postselection=self.config.get("selection", None),
        # )
        return df

    @classmethod
    def __add_cached_df(cls):
        """
        dynamically add properties by caching the values
        using the appropriate functions
        for example this adds a property df_train which just gets attribute _df_train if it exists
        or calls _load_df and sets it if it doesn't exist.
        """

        for attr_name in [
            "X_test",
            "X_train",
            "y_test",
            "y_train",
            "sample_weight_train",
            "sample_weight_test",
        ]:

            def func(self, attr_name=attr_name):
                if not hasattr(self, f"_{attr_name}"):
                    self.prep_Xy()
                return getattr(self, f"_{attr_name}")

            setattr(cls, attr_name, property(func))

        for test_train in ["test", "train"]:

            def func(self, test_train=test_train):
                df_attr_name = f"df_{test_train}"
                logger.debug(
                    f"Accessing {df_attr_name} with files_key: {test_train}_files"
                )

                files_key = f"{test_train}_files"
                files = self.data_config[files_key]
                # if isinstance(files, str):
                #     if os.path.isfile(files):
                #         files = [files]
                #     if os.path.isdir(files):
                #         files = glob.glob(files+"/{test_train}.*")
                #     if "/"
                #     elif files.startswith("tfxkit"):
                #         files

                logger.debug(f"?? Loading {df_attr_name} from files: {files}")
                if not hasattr(self, f"_{df_attr_name}"):
                    attr = self._load_df(files)
                    # assert False, f"{df_attr_name} is not set correctly {test_train}"
                    setattr(self, f"_{df_attr_name}", attr)
                return getattr(self, f"_{df_attr_name}")

            logger.debug(f"Adding property df_{attr_name} to DataManager")
            setattr(cls, f"df_{test_train}", property(func))

    def prep_Xy(self):

        if hasattr(self, "_X_train"):
            return

        xy_maker = import_function(self.data_config.xy_maker)
        logger.debug(f"Using xy_maker function: {xy_maker.__name__}")

        weight_column_train = self.data_config.get("weights_column", None)
        weight_column_test = weight_column_train
        if not weight_column_train:
            weight_column_train = self.data_config.get("weight_column_train", None)
            weight_column_test = self.data_config.get("weight_column_test", None)
        else:
            if getattr(self.data_config, "weight_column_train") or getattr(
                self.data_config, "weight_column_test"
            ):
                logger.warning(
                    "Both 'weights' and 'weight_column_train'/'weight_column_test' are set. "
                    "Using 'weights' for both train and test sets."
                )
                raise ValueError(
                    "Please specify either 'weights' or 'weight_column_train'/'weight_column_test', not both."
                )
        logger.info(
            f"Using the following weights column:\n\t\
                Training: {weight_column_train}\n\t\
                Test:     {weight_column_test}"
        )
        X_train, y_train, sample_weight_train = xy_maker(
            self.df_train,
            self.data_config.features,
            self.data_config.labels,
            weight_column_train,
        )
        X_test, y_test, sample_weight_test = xy_maker(
            self.df_test,
            self.data_config.features,
            self.data_config.labels,
            weight_column_test,
        )

        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        self._sample_weight_train = sample_weight_train
        self._sample_weight_test = sample_weight_test

        self.train = DatasetSplit(
            df=self.df_train, X=X_train, y=y_train, sample_weight=sample_weight_train
        )
        self.test = DatasetSplit(
            df=self.df_test, X=X_test, y=y_test, sample_weight=sample_weight_test
        )

        # return X_train, X_test, y_train, y_test

    def prepare_datasets(self):
        """Convert raw data into tf.data.Dataset objects."""
        pass
