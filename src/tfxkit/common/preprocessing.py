
pdg_map = {2212: 0, 1000020040: 1, 1000070140: 2, 1000130270: 3, 1000260560: 4}

pdg_labels = {
    2212: "p",
    1000020040: "He",
    1000070140: "N",
    1000130270: "AL",
    1000260560: "FE",
}

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
