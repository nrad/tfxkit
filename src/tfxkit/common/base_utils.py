import logging
from copy import deepcopy
from pprint import pprint
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


##
## Utility functions for data selection and HDF5 file reading
##

def get_selection_dict(df, class_name="filtered"):
    counts = df.groupby(class_name).apply(lambda x: len(x)).to_dict()

    min_count = min(counts.values())
    min_idx = list(counts.values()).index(min_count)

    return {f"{class_name}=={k}": round(min_count / v, 5) for k, v in counts.items()}

# def combine_weight_columns(df, weight_columns):
#     """
#     combine the weight columns into a single column
#     weight_columns can be a list of columns or a single column
#     """
#     if weight_columns is None:
#         return df.apply(lambda x: 1, axis=1)
#     weight_columns = (
#         [weight_columns] if isinstance(weight_columns, str) else weight_columns
#     )
#     return df[weight_columns].prod(axis=1)

def combine_weight_columns(df, weight_columns):
    """
    Combine weight columns and/or scalar constants into a single weight Series.
    weight_columns can be:
    - None → returns all ones
    - a single column name (str)
    - a list of column names and/or floats (e.g., [2.0, "weight"])
    """
    if weight_columns is None:
        return pd.Series(1.0, index=df.index)

    if isinstance(weight_columns, str):
        weight_columns = [weight_columns]
    
    if isinstance(weight_columns, (int, float)):
        weight_columns = [weight_columns]

    weights = pd.Series(1.0, index=df.index)
    for w in weight_columns:
        if isinstance(w, (int, float)):
            weights *= w
        else:
            weights *= df[w]

    return weights

##
## HDF5 file reading utilities
## 

def _read_hdf_in_chunks(file_path, chunksize, **kwargs):
    """
    Generator function that reads an HDF5 file in chunks and yields each chunk.
    If the last chunk is smaller than the chunk size, it is merged with the previous chunk.

    Parameters:
    - file_path: str, the path to the HDF5 file.
    - chunksize: int, the number of rows per chunk.

    Yields:
    - DataFrame chunks, where the last chunk is merged with the previous one if it's smaller than the chunksize.
    """
    last_chunk = None
    dfs = pd.read_hdf(file_path, chunksize=chunksize, **kwargs)
    print(f"DEBUG: {chunksize = }")
    for df in dfs:
        if last_chunk is not None:
            # If the current chunk is smaller than the chunk size, merge it with the last chunk
            if len(df) < chunksize:
                last_chunk = pd.concat([last_chunk, df])
                continue  # Skip the yield to process the next chunk or end the loop
            else:
                # Yield the last_chunk if the current chunk is not smaller than the chunk size
                yield last_chunk
        last_chunk = df

    # After the loop, yield the last chunk if it exists
    if last_chunk is not None:
        yield last_chunk

def read_csv(file_paths):
    file_paths = [file_paths] if isinstance(file_paths, (str)) else file_paths
    return pd.concat([pd.read_csv(f) for f in file_paths])


def read_hdfs(file_paths, chunksize=None, **kwargs):
    if chunksize is None:
        return pd.concat([read_hdf(f, **kwargs) for f in file_paths])
    else:
        return read_hdf_in_chunks(file_paths, chunksize=chunksize, **kwargs)

def read_hdf_in_chunks(file_paths, chunksize=None, **kwargs):
    """
    Generator function that reads multiple HDF5 files in chunks and yields each chunk.
    If the last chunk is smaller than the hunk size, it is merged with the previous chunk.

    Parameters:
    - file_paths: list of str, the paths to the HDF5 files.
    - chunksize: int, the number of rows per chunk.

    Yields:
    - DataFrame chunks, where the last chunk is merged with the previous one if it's smaller than the chunksize.
    """
    file_paths = [file_paths] if isinstance(file_paths, (str)) else file_paths
    # if chunksize is None:
    #     print('here')
    #     return pd.concat([pd.read_hdf(f) for f in file_paths])

    for file_path in file_paths:
        for chunk in _read_hdf_in_chunks(file_path, chunksize, **kwargs):
            yield chunk


def read_hdf(
    fname,
    selection_dict={},
    balance_by=False,
    preselection=None,
    postselection="",
    # return_selection_weight = False
    include_selection_weight=True,
    **kwargs,
):
    """
    helper function for reading hdf file
    to query the DF, and the value will be used as a fraction

    <balance_by> : if passed, it should be the name of the column to be interpreted as a class name
                 to balance the populations (see get_selection_dict)

    <selection_dict> : if passed, each key will be used to select the given fraction of the data,
                 e.g. {'filtered==1': 0.5, 'filtered==0': 0.5} will select 50% of the data for each class

    <preselection> : selection applied *before* the sample is balanced
    <postselection> : selection applied *after* the sample is balanced
    """
    kwargs = deepcopy(kwargs)
    # kwargs.setdefault('where', preselection )
    try:
        df = pd.read_hdf(fname, where=preselection, **kwargs)
    except ValueError:
        df = pd.read_hdf(fname, **kwargs)
        if preselection:
            df = df.query(preselection)

    if balance_by:
        if selection_dict:
            raise ValueError(
                "either balance_by should be given or selection_dict, not both!"
            )
        selection_dict = get_selection_dict(df, class_name=balance_by)

    if selection_dict:
        dfs = {}

        len_df = len(df)
        print(len_df)
        for sel, frac in selection_dict.items():
            df_ = df.query(sel)  # .sample(frac)
            dfs[sel] = df_

        del df
        sum_lengths = sum(map(len, dfs.values()))
        sum_values = sum(selection_dict.values())
        print(sum_lengths, sum_values)

        selection_weights = {}
        for sel, frac in selection_dict.items():
            if frac > 1:
                raise ValueError(
                    "Fractions given as values in selection_dict must be smaller than 1."
                )
            n = int(len(dfs[sel]) * frac)
            print(f"{sel}: selecting {n} out of {len(dfs[sel])}")
            # if frac and frac < 1:
            dfs[sel] = dfs[sel].sample(n)
            selection_weights[sel] = 1.0 / frac if frac else 0
        # print(dfs)
        df = pd.concat(dfs.values()).sample(frac=1)

        if include_selection_weight:
            weight_col = (
                "selection_weights"
                if not isinstance(include_selection_weight, str)
                else include_selection_weight
            )
            df[weight_col] = 0
            for sel, weight in selection_weights.items():
                df.loc[df.query(sel).index, weight_col] = weight

        assert df.index.is_unique, df[df.index.duplicated()]
    if postselection:
        df = df.query(postselection)
    return df


##
## Utility functions
##


def import_function(fn_path, strict=True):
    import importlib
    """
    Load a function from a given module path.
    Parameters:
    - fn_path: str, the full path to the function in the format 'module.submodule.function_name'.
    Returns:
    - function: the loaded function.
    Raises:
    - ValueError: if the function is not found or is not callable.
    """
    
    module_path, fn_name = fn_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    if not hasattr(module, fn_name):
        raise ValueError(f"Function {fn_name} not found in module {module_path}")
    fn = getattr(module, fn_name)
    if not hasattr(fn, "__call__") and strict:
        logger.debug(f"Loaded function {fn_name} from {module_path} but it's not callable:\n{fn}")
        raise ValueError(f"{fn_name} is not a callable function")
    logger.info(f"Loaded function {fn_name} from {module_path}")

    return fn