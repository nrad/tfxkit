import warnings
import yaml
import glob
from copy import deepcopy
import re
import pickle
import os
import pylab as plt
import inspect
import uuid
import itertools
from pprint import pprint
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

# import seaborn as sns
# import matplotlib.pyplot as plt

# import seaborn as sns
# import sklearn
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix


# from defaults import DEFAULT_DATA_COLUMNS

##
##
##


class Dumper(yaml.Dumper):
    def increase_indent(self, flow=False, *args, **kwargs):
        return super().increase_indent(flow=flow, indentless=False)


def yaml_dump(di, fname, **kwargs):
    yaml.dump(di, open(fname, "w"), Dumper)


##
# Parser
##


def get_specified_args(args, parser):
    default_args = {v.dest: v.default for k, v in parser._option_string_actions.items()}
    specified = {
        arg: value
        for arg, value in args.__dict__.items()
        if arg in default_args and value != default_args[arg]
    }
    # value != parser._option_string_actions[arg].default}
    return specified


##
##
##


def combine_weight_columns(df, weight_columns):
    """
    combine the weight columns into a single column
    weight_columns can be a list of columns or a single column
    """
    if weight_columns is None:
        return df.apply(lambda x: 1, axis=1)
    weight_columns = (
        [weight_columns] if isinstance(weight_columns, str) else weight_columns
    )
    return df[weight_columns].prod(axis=1)


def get_hostname():
    import os

    hostname = os.getenv("HOSTNAME", "")
    print(f"hostname: {hostname}")
    return hostname


##
# ils
##


def split_list_in_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def natural_sort(l):
    """stolen from:
    https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort
    """

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(l, key=alphanum_key)


def unique(sequence):
    """
    remove dublicates from a list while preserving the order
    """
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def get_file_list(pattern_or_list, strict=False, **format_kwargs):
    ret = []
    if isinstance(pattern_or_list, (tuple, list)):
        for pattern in pattern_or_list:
            pattern = pattern.format(**format_kwargs)
            if "*" in pattern:
                ret.extend(get_file_list(pattern))
            else:
                if strict and not os.path.isfile(pattern):
                    raise ValueError(f"File doesn't exist: {pattern=}")
                ret.append(pattern)
    else:
        ret.extend(natural_sort(glob.glob(pattern_or_list)))

    return ret


def split_file_list(file_list, val_frac=0.2, test_frac=0.1):
    if val_frac + test_frac > 1:
        raise ValueError("val_frac + test_frac must be less than 1")

    train_frac = 1 - val_frac - test_frac

    total_files = len(file_list)
    train_end = round(total_files * train_frac)
    val_end = train_end + round(total_files * val_frac)

    # assert train_end + val_end <= total_files
    if val_end > total_files:
        raise ValueError(
            "train_end + val_end ({}) must be less than or equal to total_files ({})".format(
                train_end, val_end, total_files
            )
        )

    print(train_frac, val_frac, test_frac)
    print(train_frac * total_files, val_frac * total_files, test_frac * total_files)
    print(train_end, val_end)

    # assert
    train_files = file_list[:train_end]
    val_files = file_list[train_end:val_end]
    test_files = file_list[val_end:]

    return train_files, val_files, test_files


##
def get_selection_dict(df, class_name="filtered"):
    counts = df.groupby(class_name).apply(lambda x: len(x)).to_dict()

    min_count = min(counts.values())
    min_idx = list(counts.values()).index(min_count)

    return {f"{class_name}=={k}": round(min_count / v, 5) for k, v in counts.items()}


def _read_hdf_in_chunks(file_path, chunksize, **kwargs):
    """
    Generator function that reads an HDF5 file in chunks and yields each chunk.
    If the last chunk is smaller than the hunk size, it is merged with the previous chunk.

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


def read_hdfs(file_paths, chunksize=None, **kwargs):
    if chunksize is None:
        return pd.concat([read_hdf(f, **kwargs) for f in file_paths])
    else:
        return read_hdf_in_chunks(file_paths, chunksize=chunksize, **kwargs)


# def read_hdfs_in_chunks(file_paths, chunksize, **kwargs):
#     """
#     Generator function that reads multiple HDF5 files in chunks and yields each chunk.
#     It patches the end of one file with the beginning of the next file,
#     ensuring only the very final chunk of all files may be smaller than the requested chunk size.

#     Parameters:
#     - file_paths: list of str, the paths to the HDF5 files.
#     - chunksize: int, the number of rows per chunk.

#     Yields:
#     - DataFrame chunks, where the last chunk of all files is merged with the previous one if it's smaller than the chunksize.
#     """
#     last_chunk = None
#     for file_path in file_paths:
#         dfs = pd.read_hdf(file_path, chunksize=chunksize, **kwargs)
#         for df in dfs:
#             if last_chunk is not None:
#                 # Merge the last chunk with the current chunk
#                 df = pd.concat([last_chunk, df])

#             if len(df) < chunksize:
#                 # If the merged chunk is still smaller than the chunksize, update last_chunk and skip the yield
#                 last_chunk = df
#                 continue
#             else:
#                 # Find out how many full chunks we can yield
#                 num_rows = len(df)
#                 full_chunks_count = num_rows // chunksize

#                 for i in range(full_chunks_count):
#                     yield df[i * chunksize : (i + 1) * chunksize]

#                 # Prepare the remainder for the next iteration or the final yield if it's the last file
#                 last_chunk = df[full_chunks_count * chunksize :]

#     # After all files have been processed, yield the last chunk if it exists
#     if last_chunk is not None:
#         yield last_chunk


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


def print_combined_df_summary(df, label="triggered"):
    n_triggered = df[label].sum()
    n_generated = len(df.triggered)
    print(
        f"""
    n({label}) = {n_triggered}
    n(generated) = {n_generated}
    n({label})/n (gen) = {n_triggered/n_generated*100:0.2f}%
    
    """
    )


def get_name_from_dict(di, sep="="):
    name_list = []
    for k, v in di.items():
        if isinstance(v, (str, int, float, type(None), bool)):
            nice_v = str(v)
        elif isinstance(v, (list, tuple)):
            nice_v = "_".join(v)
        elif isinstance(v, (dict)):
            nice_v = get_name_from_dict(v, sep=sep)
        else:
            raise Exception("not sure how to make %s nice" % v)
        if len(nice_v) > 120:
            nice_v = hash_objects(nice_v)
        if not nice_v:
            nice_v = "None"
        name_list.append(f"{str(k).replace('_','')}{sep}{nice_v}")
        # print(name_list)
    return "_".join(name_list)


def hash_objects(*args, short=True):
    hash_ = hex(sum(map(hash, args)))
    if short:
        hash_ = hash_[3:11]
    return hash_


def unique_name(prefix="tmp", suffix=None, length=None):
    hsh = uuid.uuid4().hex.replace("-", "")
    if length:
        hsh = hsh[:length]
    return "%s_%s" % (prefix, hsh)


def filter_kwargs(func, default_kwargs={}, **kwargs):
    """
    select only kwargs which are arguments for func.
    keys in default_kwargs are replaced by existing ones in kwargs
    """
    func_args = inspect.getfullargspec(func).args
    filtered_kwargs = dict(default_kwargs)
    filtered_kwargs.update(**kwargs)
    filtered_kwargs = {k: v for k, v in filtered_kwargs.items() if k in func_args}
    return filtered_kwargs


##
# Histogram Helpers
##


def plot_ratio(
    hist_dicts,
    gridspec_kw={"hspace": 0.1, "height_ratios": [3, 1]},
    ratio_idx=[],
    **common_kwargs,
):
    """
    hist_dicts = [ dict(x=hist, **hist_kwargs), ... ]
    """

    fig, (ax, axr) = plt.subplots(2, sharex=True, gridspec_kw=gridspec_kw)

    hist_common_kwargs = dict(histtype="step")
    hist_common_kwargs.update(**common_kwargs)

    res = []
    for hist_dict in hist_dicts:
        hist_kwargs = {}
        hist_kwargs.update(**hist_common_kwargs)
        hist_kwargs.update(**hist_dict)
        res.append(
            ax.hist(
                **hist_kwargs,
            )
        )

    ratio_kwargs = dict(marker="o", ms=3, linestyle="")
    ratios = []

    ratio_idx = (
        [(i, 0) for i in range(1, len(res))] if not len(ratio_idx) else ratio_idx
    )

    for nom_idx, denom_idx in ratio_idx:
        bin_cont_nom, bin_edges, _ = res[nom_idx]
        bin_cont_denom, bin_edges_nom, _ = res[denom_idx]

        ratio = bin_cont_nom / bin_cont_denom
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        axr.plot(bin_centers, ratio, **ratio_kwargs)

    ax.legend(loc="best")
    axr.grid(axis="y")

    return fig, (ax, axr)


##
# Helpers for Preprocessing
##


pdg_labels = {
    2212: "p",
    1000020040: "He",
    1000070140: "N",
    1000130270: "AL",
    1000260560: "FE",
}

pdg_map = {2212: 0, 1000020040: 1, 1000070140: 2, 1000130270: 3, 1000260560: 4}


def get_table_keys(fname):
    import tables

    with tables.open_file(fname) as f:
        return [k for k in f.root.__members__ if not k.startswith("__")]


def get_table(
    fname,
    keys=None,
    index_cols=["Run", "Event", "SubEvent", "SubEventStream", "exists"],
    key_columns={},
    add_column_prefix=True,
):
    import tables

    keys = keys if keys else get_table_keys(fname)

    dfs = {}
    with tables.open_file(fname) as f:
        for key in keys:
            if key in key_columns:
                columns_to_get = index_cols + key_columns[key]
            else:
                columns_to_get = None
            df_ = pd.DataFrame(
                data=getattr(f.root, key).read(),
                columns=columns_to_get,
            ).set_index(index_cols, drop=True)
            dfs[key] = df_
        # dfs = {key: pd.DataFrame(getattr(f.root, key).read()).set_index(index_cols, drop=True) for key in keys}

    for k, df_ in dfs.items():
        if len(df_.columns) > 1:
            if add_column_prefix:
                df_.columns = [f"{k}_{col}" for col in df_.columns]
            # else:
            #     df_.columns = []
        elif len(df_.columns) == 1 and df_.columns[0] == "value":
            df_.columns = [k]
        else:
            raise ValueError(f"unexpected single column: {df_.columns}")
    return pd.concat(dfs.values(), axis=1).reset_index()


def get_df(fname, key, columns=None, strict=True):
    import tables

    with tables.open_file(fname) as f:
        if not key in f.root.__members__:
            raise KeyError(
                f"Requested key ({key}) was not found among {f.root.__members__}"
            )

        arr = getattr(f.root, key)
        arr_read = arr.read()

        try:
            return pd.DataFrame(arr_read, columns=columns)
        except ValueError:
            scalar_cols = []
            vector_cols = []

            col_names = arr.colnames
            for col_name in col_names:
                col_shape = arr_read[col_name].shape
                if len(col_shape) == 1:
                    scalar_cols.append(col_name)
                else:
                    vector_cols.append((col_name, col_shape[1]))
            print(scalar_cols)
            print(vector_cols)

            df_scalar = pd.DataFrame(arr_read[scalar_cols])

            df_vector = []
            for col_name, col_size in vector_cols:
                # col_name = 'energy'
                # n_ar_read = ar_read[col_name].shape[1]
                subcol_names = [f"mu{idx}_{col_name}" for idx in range(1, col_size + 1)]
                df_vector.append(pd.DataFrame(arr_read[col_name], columns=subcol_names))

            return pd.concat([df_scalar] + df_vector, axis=1)


def extract_pdg(x):
    """
    Extracts the atomic number (Z) and atomic mass (A) from PDG.
    (atomic PDG is given by 10LZZZAAAI)
    return Z, A
    """

    pdg = int(x)
    # return pdg

    if pdg == 2212:
        Z, A = (1, 1)
    elif pdg // 1e8 == 10:
        Z = (pdg % 1e7) // 1e4  # extract digits 5 to 7
        A = (pdg % 1e4) // 1e1  # extract digits 2 to 4
    else:
        raise NotImplementedError("could not extract atomic number for %s" % pdg)
    # print(Z,A)
    return {"Z": Z, "A": A}


# pdg = df['pdg_encoding']
def get_Z_from_pdg(pdg):
    """
    extracts atomic number (Z) from pdg assuming PDG format 10LZZZAAAI
    """
    Z = (pdg % 1e7) // 1e4  # extract digits 5 to 7
    Z[pdg == 2212] = 1
    return Z


def get_A_from_pdg(pdg):
    """
    extracts atomic mass (A) from pdg assuming PDG format 10LZZZAAAI
    """
    A = (pdg % 1e4) // 1e1  # extract digits 2 to 4
    A[pdg == 2212] = 1
    return A


##########################################################################
##########################################################################


def combine_and_preproc(
    fnames,
    output=None,
    preselection="",
    postselection="",
    balance_by="filtered",
    do_preproc=True,
    data_columns=["multiplicity", "n_photons"],
    selection_dict={},
    include_selection_weight="selection_weights",
    pre_combine_func=None,
):
    dfs = []
    for fname in fnames:
        df = read_hdf(
            fname,
            balance_by=balance_by,
            preselection=preselection,
            postselection=postselection,
            selection_dict=selection_dict,
            include_selection_weight=include_selection_weight,
        )
        if pre_combine_func:
            if not callable(pre_combine_func):
                raise ValueError("pre_combine_func should be a function, not a string")
            print("!! Applying Pre-combine function", "df_shape before =", df.shape)
            df = pre_combine_func(df)
            print("!!                                  df_shape after =", df.shape)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True).sample(frac=1)

    if do_preproc:
        preproc_func = do_preproc if callable(do_preproc) else preproc
        df = preproc_func(df)
    if output:
        extention = output.split(".")[-1]
        if not extention:
            raise ValueError("output file should have an extention")

        if extention in ["h5", "hdf", "hdf5"]:
            df.to_hdf(
                output,
                key="combined",
                format="table",
                data_columns=data_columns,
            )
        if extention in ["parquet"]:
            df.to_parquet(output)
    else:
        return df


# def combine_and_preproc_inchunks(fnames, output=None, preselection="", balance_by="filtered", output_max_size=3):
#     dfs = []
#     sizes = []
#     for fname in fnames:
#         df = read_hdf(fname, balance_by=balance_by, preselection=preselection)
#         sizes.append( df.memory_usage()*1E-9 )

#         dfs.append(df)

#     df = preproc(pd.concat(dfs, ignore_index=True).sample(frac=1))
#     if output:
#         df.to_hdf(output, key="combined", format="table")
#     else:
#         return df


def combine_and_preproc_inchunk(
    fnames, output=None, preselection="", balance_by="filtered", max_size=5e8
):
    # raise NotImplementedError()
    # Initialize an empty DataFrame
    combined_df = pd.DataFrame()
    chunk_num = 0

    sizes = []

    for fname in fnames:
        df = read_hdf(fname, balance_by=balance_by, preselection=preselection)
        size = df.memory_usage().sum() * 1e-9
        sizes.append(size)
        # Check if adding this DataFrame will exceed the memory limit
        if sum(sizes) >= max_size:
            if output:
                combined_df = preproc(combined_df.sample(frac=1))

                combined_df.to_hdf(
                    output + f"{chunk_num:02g}", key=f"combined", format="table"
                )
                chunk_num += 1
                combined_df = df  # Start a new DataFrame for the next chunk
            else:
                raise MemoryError(
                    "DataFrame size exceeds memory limit and no output file specified"
                )
        else:
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Preprocess and write/shuffle the remaining DataFrame if it's not empty
    if not combined_df.empty:
        combined_df = preproc(combined_df.sample(frac=1))
        if output:
            combined_df.to_hdf(output, key=f"combined_{chunk_num}", format="table")
        else:
            return combined_df


# Usage
# combine_and_preproc(fnames, output="output_file.h5", max_memory=500000000) # Example usage
def get_weights(
    files, nfiles=None, flux="GaisserH4a", return_weighter=False, verbose=True
):
    import simweights

    flux = getattr(simweights, flux)() if isinstance(flux, str) else flux

    if verbose:
        print(f"{simweights.__path__=}")
        print(f"{simweights.__version__=}")
    files = [files] if isinstance(files, str) else files
    weighters = []
    for f in files:
        hdffile = pd.HDFStore(f, "r")
        weighter = simweights.CorsikaWeighter(hdffile, nfiles=nfiles)
        weighters.append(weighter)

    weighter = sum(weighters)
    weights = weighter.get_weights(flux)

    if verbose:
        print(weighter.tostring(flux))
    if return_weighter:
        return weights, weighter
    else:
        return weights


def add_elbert_yield(df):
    from utils.elbert_yield import ElbertYield

    A = get_A_from_pdg(df["pdg_encoding"])

    elbert = ElbertYield(A=A, primary_energy=df["energy"], cos_theta=df["cos_theta"])

    x_mu = df["shower_mu1_energy"] / (df["energy"] / A)
    df["x_mu"] = x_mu
    df["elbert_yield"] = elbert.get_yield(x_mu)
    df["elbert_conv"] = elbert.get_conventional_yield(x_mu)
    df["elbert_prompt"] = elbert.get_prompt_yield(x_mu)
    df["elbert_prefactor_conv"] = elbert._compute_prefactor(False)[0]
    df["elbert_prefactor_prompt"] = elbert._compute_prefactor(True)[0]
    df["elbert_prob"] = 1 - elbert.get_prob(x_mu)

    df.fillna(0, inplace=True)
    df.replace([float("inf"), -float("inf")], 0, inplace=True)
    return df


def preproc(df, query=None):
    if "log_energy" in df.columns:
        df = add_elbert_yield(df)
        return df

    if query:
        df = df.query(query)

    decode_pdg = False
    A = get_A_from_pdg(df["pdg_encoding"])
    df["atomic_mass"] = A

    log = np.log10

    if decode_pdg:
        Z = get_Z_from_pdg(df["pdg_encoding"])
        # A = get_A_from_pdg(df["pdg_encoding"])
        df["atomic_mass"] = A
        df["atomic_number"] = Z
        df["target"] = df["pdg_encoding"].apply(lambda x: pdg_labels[x])

    # df['log_energy'] = log(df['energy'])
    df["log_length"] = log(df["length"].replace(0, np.nan))
    # for col in df.columns:
    #    if "energy" in col:
    #        log_col_name = col.replace("energy", "log_energy")
    #        if log_col_name not in df.columns:
    #            df[log_col_name] = log(df[col].replace(0, np.nan))

    mu_vars = ["energy", "radius"]
    n_mu = 10

    mu_pattern = "mu(\d+)_"
    mu_vars_all = [v for v in df.columns if re.match(mu_pattern, v)]
    mu_vars = list(set([x.split("_", 1)[-1] for x in mu_vars_all]))
    print(f"{mu_vars_all = }")
    print(f"{list(df)}")
    n_mu = max(int(re.search(mu_pattern, var).group(1)) for var in mu_vars_all)
    # mu_energy_columns = [x for x in mu_vars_all if "energy" in x]
    mu_energy_columns = [
        mu_pattern.replace("(\d+)", str(i)) + "energy" for i in range(1, n_mu + 1)
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        for mu_var in mu_vars:
            var_list = ["mu%s_%s" % (imu, mu_var) for imu in range(1, n_mu + 1)]
            df.loc[:, "mu_%s_mean" % mu_var] = np.nanmean(
                df[var_list].replace(0, np.nan), axis=1
            )
            df.loc[:, "mu_%s_std" % mu_var] = np.nanstd(
                df[var_list].replace(0, np.nan), axis=1
            )
            df.loc[:, "mu_%s_max" % mu_var] = np.nanmax(
                df[var_list].replace(0, np.nan), axis=1
            )
            df.loc[:, "mu_%s_min" % mu_var] = np.nanmin(
                df[var_list].replace(0, np.nan), axis=1
            )

    # df.loc[:, "bundle_energy"] =
    # mu_bundle_energy
    # df.loc[:, 'max_fractional_energy'] = df['
    df.loc[:, "energy_per_nucleon"] = df["energy"] / A
    df.loc[:, "mu_bundle_energy"] = df[mu_energy_columns].sum(axis=1)

    for col in df.columns:
        if "energy" in col:
            log_col_name = col.replace("energy", "log_energy")
            if log_col_name in df.columns:
                raise RuntimeError(
                    f"column {log_col_name = } already existed in the dataframe... probably running preproc twice! The DF has the columns: {df.columns = }"
                )
            df.loc[:, log_col_name] = log(df[col].replace(0, np.nan))

    df["rho"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
    df["r"] = np.sqrt(df["rho"] ** 2 + df["z"] ** 2)
    for col in ["z", "rho", "r", "multiplicity"]:
        df.loc[:, "log_%s" % col] = log(df[col].replace(0, np.nan))

    df["pdg_map"] = df["pdg_encoding"].apply(lambda x: pdg_map[x])

    # df.loc[:, 'normalized_energy'] = df['energy']/A
    # df.loc[:, "mu_bundle_log_energy"] = log(df["mu_bundle_energy"])

    df.loc[:, "mu_bundle_energy_fraction"] = df["mu_bundle_energy"] / (df["energy"] / A)
    df.loc[:, "mu_leading_energy_fraction"] = df["mu_energy_max"] / (df["energy"] / A)
    df.loc[:, "singleness"] = df["mu_energy_max"] / (
        df["mu_bundle_energy"]
    )  # max_fractional_energy

    # df["zenith"] = np.arccos(df["z"] / df["r"])
    # df["azimuth"] = np.arctan2(df["y"], df["x"]) + np.pi

    if "azimuth" in df.columns:
        df.loc[:, "cos_azimuth"] = np.cos(df["azimuth"])
    if "zenith" in df.columns:
        df.loc[:, "cos_zenith"] = np.cos(df["zenith"])

    df = df.fillna(0)

    return df


def get_true_frac(y):
    try:
        y = y.to_numpy()
    except AttributeError:
        pass
    return (y == 1).sum() / len(y)


def get_balanced_df(df, by="triggered"):
    """
    will return a df with balanced population by undersampling majority class
    """
    counts = df[by].value_counts()
    print("counts:", counts)
    sample_size = min(counts)
    sampled_df = (
        df.groupby([by]).apply(lambda x: x.sample(sample_size)).reset_index(drop=True)
    ).sample(frac=1)
    return sampled_df


def get_balanced_mask(y, frac=0.4):
    """
    will return a mask which throws away <frac> of cases where y=False
    """
    try:
        y = y.to_numpy()
    except AttributeError:
        pass

    not_triggered = y == 0
    triggered = y == 1  #
    mask_random = np.random.random(size=len(not_triggered)) < frac
    mask_balanced = np.where(not_triggered, mask_random, triggered)
    return mask_balanced


def label_sim_levels(
    df, sim_levels=["generated", "triggered", "filtered", "Level3"], key="sim_level"
):
    masks = {}
    for i, t in enumerate(sim_levels):
        query_this = f"{t}==1"
        query_rest = [
            f"{t_}==0"
            for t_ in sim_levels
            if sim_levels.index(t_) > sim_levels.index(t)
        ]
        query = " & ".join([query_this] + query_rest)
        mask = df.query(query).index
        print(query)
        masks[t] = mask
        df.loc[mask, key] = t
    return df, masks


def apply_selection_fractions(
    df,
    selection_fractions,
    selection_keys=["generated", "triggered", "filtered", "Level3"],
    include_selection_weight="selection_weights",
):

    keep_prob = pd.Series(0, index=df.index)
    for k in selection_keys:
        selection_fractions.setdefault(k, 1)

    values = [selection_fractions[k] for k in selection_keys]

    if not values == list(sorted(values)):
        raise ValueError(
            "selection_fractions must be ordered in decreasing order of selection. %s"
            % values
        )

    for k in selection_keys:
        keep_prob.loc[df[k].astype(bool)] = selection_fractions[k]

    mask = np.random.random(size=len(df)) < keep_prob

    df = df[mask]

    if include_selection_weight:
        if include_selection_weight in df.columns:
            raise ValueError(
                f"Column {include_selection_weight} already exists in df. Please remove it first or use a different name"
            )
        df.loc[:, include_selection_weight] = 1.0 / keep_prob[mask]

    return df


###################################
##       useful functions:
##################################


def run_func_in_paral(func, args, n_proc=None, starmap=False):
    """
    will have the same output as list(map(func, args))
    but it will run <n_proc> number of parallel jobs

    this will fail if the <func> is not picklable
    """
    import multiprocessing

    n_cpus = multiprocessing.cpu_count()
    if n_proc:
        if n_proc > n_cpus:
            print(
                f"WARNING... number of requested processes ({n_proc}) larger than cpus ({n_cpus})....this will make someone angry..."
            )
            n_proc = n_cpus
    else:
        print(f"... setting n_proc to {n_cpus}")
        n_proc = n_cpus

    if n_proc > 1:
        pool = multiprocessing.Pool(processes=n_proc)
        if starmap:
            results = pool.starmap(func, args)
        else:
            results = pool.map(func, args)
        pool.close()
        pool.join()
    else:
        if starmap:
            from itertools import starmap

            results = starmap(func, args)
        else:
            results = list(map(func, args))
    return results


##
# Helpers for validation
##
