import tables
import pandas as pd
import numpy as np
from utils import get_df, get_weights, preproc, get_table_keys, get_table
import utils
# DEFAULT_FILTER_NAMES, DEFAULT_DATA_COLUMNS, DEFAULT_CSCD_KEYS
from defaults import default_config

# def get_df(fname, key, columns=None):
#    with tables.open_file(fname) as f:
#        try:
#            return pd.DataFrame( getattr(f.root, key).read() , columns=columns)
#        except KeyError:
#            raise KeyError("key not found")


columns = ["x", "y", "z", "length", "pdg_encoding", "energy"]


def combine_gen_det(
    fgen,
    fdet,
    ffilt=None,
    fl3=None,
    fcscdbdt=None,
    columns=columns,
    key="PolyplopiaPrimary",
    extra_keys=default_config['filter_names']+["ProcessingWeight"],
    # extra_keys= DEFAULT_CSCD_KEYS,
    index=["energy", "minorID"],
    filter_names=[],
    gen_weights_nfiles=0,
    preproc=False,
    selection_fractions={},
    selection_weight_key="selection_weights",
):
    """
    Combines the detector and generated level files according to the given indices.

    Parameters:
    - fgen (str): Path to the generated level file.
    - fdet (str): Path to the detector level file.
    - ffilt (str, optional): Path to the filter level file. Default is None.
    - fl3 (str, optional): Path to the Level3 file. Default is None.
    - columns (list, optional): List of columns to include in the combined dataframe. Default is columns.
    - key (str, optional): Key to use for merging the files. Default is "PolyplopiaPrimary".
    - extra_keys (list, optional): List of additional keys to include in the combined dataframe. Default is ["muons"].
    - index (list, optional): List of keys to use as the index for the combined dataframe. Default is ["energy", "minorID"].
    - filter_names (list, optional): List of filter names to include in the combined dataframe. Default is an empty list.
    - gen_weights_nfiles (int, optional): Number of files to use for generating weights. Default is 0.
    - preproc (bool, optional): Whether to preprocess the combined dataframe. Default is False.
    - selection_fractions (dict, optional): Dictionary of selection fractions to apply to the combined dataframe. Default is an empty dictionary.

    Returns:
    - df (DataFrame): Combined dataframe containing the merged information from the input files.
    """
    if not index:
        raise ValueError(
            "Index must be given as a list of keys, otherwise I cannot combine the files consistently"
        )

    columns_to_get = list(set(columns + index)) if columns else index
    gen_cols = ["Run", "Event", "SubEvent", "SubEventStream", "exists"]
    df_gen = get_df(fgen, key, columns=None)
    df_det = get_df(fdet, key, columns=columns_to_get)

    if extra_keys:
        for extra_key in extra_keys:
            if extra_key == "muons":
                df_gen_mu = get_df(fgen, extra_key)
                mu_cols = ["depth", "cos_theta"] + [
                    col for col in df_gen_mu.columns if col.startswith("mu")
                ]
                df_gen_mu = df_gen_mu[mu_cols]
                df_gen = pd.concat([df_gen, df_gen_mu], axis=1)
                del df_gen_mu

            if extra_key == "ProcessingWeight":
                try:
                    df_gen_w = get_table(fgen, [extra_key])
                    df_gen_w = df_gen_w[[extra_key]]
                    df_gen = pd.concat([df_gen, df_gen_w], axis=1)
                except:
                    print(f"WARNING - {extra_key} not found in the file")
                    df_gen_w = np.ones(len(df_gen))
                    df_gen['ProcessingWeight'] = df_gen_w

                del df_gen_w

    def set_index(df): return df.set_index(index, drop=False)
    df_gen = set_index(df_gen)
    df_det = set_index(df_det)

    df_det["triggered"] = 1
    df_gen["generated"] = 1

    if gen_weights_nfiles:
        # nfiles = gen_weights_nfiles["nfiles"]
        df_gen["flux_weights"] = get_weights(
            fgen, nfiles=gen_weights_nfiles).astype('float128')
        for flux in ["GlobalFitGST", "GaisserH4a_IT", "GaisserHillas",  "GaisserH3a"]:
            df_gen[f"{flux}_weights"] = get_weights(
                fgen, nfiles=gen_weights_nfiles, flux=flux).astype('float128')

    to_concat = [df_gen, df_det[["triggered"]]]
    if ffilt:
        df_filt = get_df(ffilt, key, columns=index + filter_names)
        df_filt = set_index(df_filt)
        df_filt["filtered"] = 1
        to_concat.append(df_filt[["filtered"] + filter_names])
        # assert len(df_filt) == len(df_det), (len(df_filt), len(df_det))
    if fl3:
        df_l3 = get_df(fl3, key, columns=index)
        df_l3 = set_index(df_l3)
        df_l3["Level3"] = 1
        to_concat.append(df_l3[["Level3"]])
        # assert len(df_l3) == len(df_det), (len(df_l3), len(df_det))

    if fcscdbdt:
        # check if empty
        try:
            df_cscdbdt = get_df(fcscdbdt, key, columns=index + ["cscd_bdt"])

            df_extra = get_table(
                fcscdbdt, keys=default_config['cascade_keys'], )
            extra_cols = list(df_extra.columns)
            df_cscdbdt = pd.concat([df_cscdbdt, df_extra], axis=1)

            df_cscdbdt = set_index(df_cscdbdt)
            df_cscdbdt["CscdBDT"] = 1
        except KeyError as e:
            print(f"WARNING - {key} not found in the file")
            print(f"WARNING - {e}")
            print(f"WARNING - creating an empty DF")
            df_cscdbdt = pd.DataFrame(columns=index + ["CscdBDT"])
            df_cscdbdt["CscdBDT"] = None
            extra_cols = default_config['cascade_keys']
            for col in extra_cols:
                df_cscdbdt[col] = None

        to_concat.append(
            df_cscdbdt[["CscdBDT"]+[k for k in extra_cols if k not in gen_cols]])

    if False:  # :)
        for df_ in to_concat:
            print("------------" * 10)
            print(df_.columns)

    df = pd.concat(to_concat, axis=1)
    df["triggered"] = df["triggered"].fillna(0)
    if ffilt:
        df["filtered"] = df["filtered"].fillna(0)
        for f in filter_names:
            df[f] = df[f].fillna(value=0)
        df["filtered"] = np.where(df["n_photons"] == 0, 0, df["filtered"])
    if fl3:
        df["Level3"] = df["Level3"].fillna(0)
    if fcscdbdt:
        df["CscdBDT"] = df["CscdBDT"].fillna(0)

    print(
        f"""
        {len(df_cscdbdt) = }
        {len(df_l3) = }
        {len(df_filt) = }
        {len(df_det) = }
        {len(df_gen) = }
        {len(df)     = } 
        """
    )

    if len(df) != len(df_gen):
        print(
            f"Some events in df_det were not in df_gen! {len(df_gen) = }, {len(df_det) = }, diff = {len(df_det) - len(df_gen)}")

        keep = False
        if keep:
            print("We will keep them anyways: ")
        # Fill the dataframe with the information from df_det
            mask_not_generated = (df["triggered"] == 1) & (
                df["generated"] != 1)
            df.loc[mask_not_generated] = df_det[mask_not_generated]
            df["generated"] = df["generated"].fillna(0)
        else:
            df = df.query("generated==1")

    selection_keys = ["generated", "triggered",
                      "filtered", "Level3", "CscdBDT"]
    if selection_fractions:
        df = utils.apply_selection_fractions(
            df,
            selection_fractions=selection_fractions,
            selection_keys=selection_keys,
            include_selection_weight=selection_weight_key,
        )

    if preproc:
        if hasattr(preproc, '__call__'):
            df = preproc(df)
        elif preproc == True:
            df = utils.preproc(df)
        else:
            raise ValueError(
                f"preproc must be a function or True, got {preproc} instead: {type(preproc)=}")
    return df


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Action

    class DictArgsAction(Action):
        """
        Custom action to parse dictionary arguments from the command line.
        """

        def __call__(self, parser, namespace, values, option_string=None):
            # Ensure the number of arguments is even
            if len(values) % 2 != 0:
                parser.error(
                    "The number of --dictargs arguments must be even, representing key-value pairs."
                )

            # Convert values to float and construct the dictionary
            it = iter(values)
            try:
                d = {k: float(v) for k, v in zip(it, it)}
            except ValueError:
                parser.error("All values must be floats.")

            setattr(namespace, self.dest, d)

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input_file_gen",
        default=None,
        help="Generated file",
    )
    parser.add_argument(
        "--input_file_det",
        default=None,
        help="Detector file",
    )
    parser.add_argument(
        "--input_file_filt",
        default=None,
        help="Filter file",
    )
    parser.add_argument(
        "--input_file_l3",
        default=None,
        help="Level3 file",
    )
    parser.add_argument(
        "--input_file_cscdbdt",
        default=None,
        help="CscdBDT file",
    )
    parser.add_argument(
        "--output_file",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--key", default="PolyplopiaPrimary")
    parser.add_argument("--columns", nargs="+", default=None)
    # parser.add_argument('--index', nargs='+', default=['energy', 'minorID'])
    parser.add_argument("--index", nargs="+", default=["energy", "minorID"])
    parser.add_argument("--gen_weights_nfiles", type=int, default=0)
    parser.add_argument("--keep_filters", action="store_true")
    parser.add_argument("--preproc", action="store_true")
    parser.add_argument(
        "--selection_fractions",
        default={},
        nargs="+",
        action=DictArgsAction,
        metavar=("KEY", "VALUE"),
    )
    parser.add_argument("--selection_weight_key", default="selection_weights")
    parser.add_argument('--add_x_mins', action='store_true')
    args = parser.parse_args()

    fgen = args.input_file_gen
    fdet = args.input_file_det
    ffilt = args.input_file_filt
    fl3 = args.input_file_l3
    fcscdbdt = args.input_file_cscdbdt
    columns = args.columns
    index = args.index
    key = args.key
    output_file = args.output_file
    filter_names = default_config['filter_names'] if args.keep_filters else []
    gen_weights_nfiles = args.gen_weights_nfiles
    # preproc = args.preproc
    selection_fractions = args.selection_fractions
    selection_weight_key = args.selection_weight_key

    preproc_funcs = []

    if args.preproc:
        preproc_funcs.append(utils.preproc)

    if args.add_x_mins:
        from utils.muon_bias_utils import get_mu_bias_from_df
        preproc_funcs.append(get_mu_bias_from_df)

    if preproc_funcs:
        print('Preprocessing functions:')
        for func in preproc_funcs:
            print(f" - {func.__name__}")

        def preproc(df):
            for func in preproc_funcs:
                df = func(df)
            return df
    else:
        preproc = None

    print(index)
    df = combine_gen_det(
        fgen,
        fdet,
        ffilt=ffilt,
        fl3=fl3,
        fcscdbdt=fcscdbdt,
        key=key,
        columns=columns,
        index=index,
        gen_weights_nfiles=gen_weights_nfiles,
        filter_names=filter_names,
        preproc=preproc,
        selection_fractions=selection_fractions,
        selection_weight_key=selection_weight_key,
    )
    if args.verbose:
        print(df.head())
        print(df.index)

    data_columns = default_config['data_columns']
    # print(f"{list(df.columns)=}")
    # print(f"{df.index=}")
    print(f"{df=}")
    print(f"{df.index=}")
    print(f"{list(df.columns)}")

    for col in df.columns:
        if list(df.columns).count(col) > 1:
            print(f"{col=}, len = {list(df.columns).count(col)}")

    df.reset_index(drop=True).sample(frac=1.0).to_hdf(
        output_file,
        key="combined",
        format="table",
        data_columns=data_columns,
    )
