import os
import numpy as np
import pandas as pd
import glob
import utils


hostname = utils.get_hostname()


if "zeuthen.desy.de" in hostname:
    filename_dir = "/lustre/fs23/group/icecube/nrad/data/zzSBU/event_selection/hdf/"
    zzSBU_dir = "/lustre/fs23/group/icecube/nrad/data/zzSBU/"
elif "icecube.wisc.edu" in hostname:
    zzSBU_dir = "/home/navidkrad/work/zzSBU/"
    filename_dir = "/home/navidkrad/work/zzSBU/event_selection/hdf/"


filename_muongun = os.path.join(filename_dir, "21315-21319_final_all.h5")
filename_data = os.path.join(filename_dir, "IC86_2010-2020_final_all_burn.h5")


keys = ["cscdSBU_MuonWeight_GaisserH4a"]
keys = [
    "cscdSBU_L4VetoTrack_cscdSBU_MonopodFit4_noDCVetoCharge",
    "cscdSBU_MonopodFit4",
    "cscdSBU_I3XYScale_noDC",
]
keys += [
    "cscdSBU_LE_bdt_cascade",
    "cscdSBU_LE_bdt_hybrid",
    "cscdSBU_LE_bdt_track",
    "cscdSBU_LE_bdt_input",
    "cscdSBU_LE_tags",
    "cscdSBU_Qtot_HLC",
]


nugen_simids = [21813, 21814, 21867, 21868, 21870, 21871]
nugen_weight_keys = [
    "I3MCWeightDict",
    "mceq_HillasGaisser2012_H4a_CORSIKA_SouthPole_December_SIBYLL2.3c_conv",
    "mceq_HillasGaisser2012_H4a_CORSIKA_SouthPole_June_SIBYLL2.3c_conv",
    "cscdSBU_HillasGaisser2012_H4a_CORSIKA_SouthPole_December_SIBYLL2.3c_conv_step1000_passing_rate",
    "mceq_HillasGaisser2012_H4a_CORSIKA_SouthPole_December_SIBYLL2.3c_pr",
    "mceq_HillasGaisser2012_H4a_CORSIKA_SouthPole_June_SIBYLL2.3c_pr",
    "cscdSBU_HillasGaisser2012_H4a_CORSIKA_SouthPole_December_SIBYLL2.3c_pr_step1000_passing_rate",
]

key_column_dict = {"I3MCWeightDict": ["PrimaryNeutrinoEnergy", "NEvents", "OneWeight"]}


def read_hdf_multiple_datasets(filename, keys=None, key_column_dict={}):
    """
    Read multiple datasets from a single HDF5 file and return them as a single dataframe.
    The function reads all keys in the file if keys is None.
    The function also renames the 'value' column to the key name.

    """
    cols_to_ignore = ["Run", "Event", "SubEvent", "SubEventStream", "exists"]

    with pd.HDFStore(filename, mode="r") as store:
        store_keys = [k.lstrip("/") for k in store.keys()]
        if keys is None:
            keys = store_keys
            print("Will read all keys: %s" % keys)
        else:
            for k in keys:
                if k not in store_keys:
                    print("Key %s not found in the file. Ignoring it." % k)
                    keys.remove(k)

        dfs = []
        found_run_info = False

        for ikey, store_key in enumerate(keys):
            key = store_key.lstrip("/")
            df = store[store_key]
            if key in key_column_dict:
                df = df[key_column_dict[key]]

            # Ignore specified columns if they exist
            # print(ikey, store_key, df.columns, found_run_info)
            if not found_run_info:
                if all([col in df.columns for col in cols_to_ignore]):
                    found_run_info = True
            else:
                df = df.drop(
                    columns=[col for col in cols_to_ignore if col in df.columns]
                )
            # Rename 'value' column to the key name
            # if len(df.columns)==1 and 'value' in df.columns:
            if "value" in df.columns:
                df = df.rename(columns={"value": key})
            # For other columns, combine key and column name to differentiate them
            df = df.rename(
                columns={
                    col: f"{key}_{col}"
                    for col in df.columns
                    if (col != key) and (col not in cols_to_ignore)
                }
            )
            dfs.append(df)

    # Concatenate all dataframes along columns
    final_df = pd.concat(dfs, axis=1)
    return final_df


##
## Nugen setup
##

nugen_nfiles = {
    21813: 9998,
    21814: 9998,
    21867: 100,
    21868: 100,
    21870: 100,
    21871: 100,
    21938: 99,
    21939: 100,
    21940: 96,
}


def add_nugen_weights(df, nfiles):
    nu_fit_params = dict(
        astro_norm=1.58,
        astro_index=-2.53,
        conv_norm=1.02,
        pr_norm=0,
    )

    def get_astro_w(df, **params):

        primary_energy = df["I3MCWeightDict_PrimaryNeutrinoEnergy"]
        one_weight = df["I3MCWeightDict_OneWeight"]
        nevents = df["I3MCWeightDict_NEvents"]

        astro_index = params.get("astro_index", nu_fit_params["astro_index"])
        astro_norm = params.get("astro_norm", nu_fit_params["astro_norm"])
        ltime = params.get("ltime", 1)
        nfiles = params.get(
            "nfiles",
        )

        flux = (
            astro_norm
            * 10 ** (-18)
            * np.power(primary_energy / 100000, astro_index)
            * ltime
        )
        return flux * one_weight / (nevents * nfiles)

    def get_conv_w(df, **params):
        one_weight = df["I3MCWeightDict_OneWeight"]
        nevents = df["I3MCWeightDict_NEvents"]

        conv_dec = df[
            "mceq_HillasGaisser2012_H4a_CORSIKA_SouthPole_December_SIBYLL2.3c_conv"
        ]
        conv_jun = df[
            "mceq_HillasGaisser2012_H4a_CORSIKA_SouthPole_June_SIBYLL2.3c_conv"
        ]
        conv_weight = (conv_dec + conv_jun) / 2
        conv_passrate = df[
            "cscdSBU_HillasGaisser2012_H4a_CORSIKA_SouthPole_December_SIBYLL2.3c_conv_step1000_passing_rate"
        ]

        conv_norm = params.get("conv_norm", nu_fit_params["conv_norm"])
        ltime = params.get("ltime", 1)
        nfiles = params.get(
            "nfiles",
        )

        return (
            one_weight
            * 2
            / (nevents * nfiles)
            * conv_norm
            * conv_weight
            * conv_passrate
            * ltime
        )

    def get_pr_w(df, **params):
        one_weight = df["I3MCWeightDict_OneWeight"]
        nevents = df["I3MCWeightDict_NEvents"]

        pr_dec = df[
            "mceq_HillasGaisser2012_H4a_CORSIKA_SouthPole_December_SIBYLL2.3c_pr"
        ]
        pr_jun = df["mceq_HillasGaisser2012_H4a_CORSIKA_SouthPole_June_SIBYLL2.3c_pr"]
        prompt_weight = (pr_dec + pr_jun) / 2
        prompt_passrate = df[
            "cscdSBU_HillasGaisser2012_H4a_CORSIKA_SouthPole_December_SIBYLL2.3c_pr_step1000_passing_rate"
        ]

        prompt_norm = params.get("pr_norm", nu_fit_params["pr_norm"])
        ltime = params.get("ltime", 1)
        nfiles = params.get(
            "nfiles",
        )
        return (
            one_weight
            * 2
            / (nevents * nfiles)
            * prompt_norm
            * prompt_weight
            * prompt_passrate
            * ltime
        )

    # df['final_astro_w'] = get_astro_w(df['I3MCWeightDict_PrimaryNeutrinoEnergy'], df['I3MCWeightDict_OneWeight'], df['I3MCWeightDict_NEvents'], nfiles)
    # df['final_conv_w'] = get_conv_w(df['I3MCWeightDict_OneWeight'], df['conv_value'], df['conv_passrate_value'], df['I3MCWeightDict_NEvents'], nfiles)
    # df['final_pr_w'] = get_pr_w(df['I3MCWeightDict_OneWeight'], df['pr_value'], df['pr_passrate_value'], df['I3MCWeightDict_NEvents'], nfiles)

    df["weights_astro"] = get_astro_w(df, nfiles=nfiles, **nu_fit_params)
    df["weights_conv"] = get_conv_w(df, nfiles=nfiles, **nu_fit_params)
    df["weights_prompt"] = get_pr_w(df, nfiles=nfiles, **nu_fit_params)


##
## Data
##


def get_livetime(year, bb="burn"):
    # grl_dir = "/Users/zhang/Google Drive/studyAndWork/IceCube/pass2_GlobalFit/good_run_list/"
    # https://wiki.icecube.wisc.edu/index.php/Pass2_Multi_Year_Cascade_Analysis#Good_Run_List
    grl_dir = f"{zzSBU_dir}/good_run_list/"
    grls = glob.glob(grl_dir + f"*{year}*.txt")
    grl = grls[0]
    df = pd.read_csv(
        grl, sep="\s+", usecols=[0, 1, 2, 3], skiprows=[1], index_col=False
    )
    if bb == "burn":
        cut = df["RunNum"] % 10 == 0
    elif bb == "blind":
        cut = ~(df["RunNum"] % 10 == 0)
    elif bb == "all":
        cut = np.ones_like(df["RunNum"])
    else:
        print("wrong bb settings")
        return False
    livetime = df["LiveTime(s)"][cut].sum()
    return livetime


def get_pass2_ltime():
    ltime_pass2_burn = 0
    for year in range(2010, 2021):
        # print(f'{year = }')
        ltime_pass2_burn += get_livetime(year, "burn")
    # for year in range(2011,2021):
    #    ltime_pass2_burn += get_livetime(year,"all")
    print(f"{ltime_pass2_burn = }")
    return ltime_pass2_burn


def apply_sel(
    df,
):
    # if True:
    #     return df
    df = df.query(
        "cos(cscdSBU_MonopodFit4_zenith)<1 & cos(cscdSBU_MonopodFit4_zenith)>-1"
    )
    df = df.query("cscdSBU_MonopodFit4_z<1000 & cscdSBU_MonopodFit4_z>-1000")
    # energy_max=10**4,energy_min=0
    df = df.query("cscdSBU_MonopodFit4_energy>0")
    # df = df.query("cscdSBU_MonopodFit4_energy<1E6")
    return df


pass2_livetime = get_pass2_ltime()

####
####
####

##
## Get Nugen Samples
##

df_nugen = {}
for simid in nugen_simids:
    # filename = f"/home/navidkrad/work/zzSBU/event_selection/hdf/{simid}_all.h5"
    filename = os.path.join(filename_dir, f"{simid}_all.h5")
    if os.path.exists(filename):
        # summary[simid]=get_nugen(simid,sel="all")
        df_nugen[simid] = read_hdf_multiple_datasets(
            filename, keys=keys + nugen_weight_keys, key_column_dict=key_column_dict
        )
        add_nugen_weights(df_nugen[simid], nfiles=nugen_nfiles[simid])
    else:
        print(f"File {filename} does not exist.")

df_nugen_all = pd.concat(df_nugen.values())

##
## Get Data
##

df_data = read_hdf_multiple_datasets(filename_data, keys=keys)
df_data = apply_sel(df_data)
df_data["weights"] = 1.0 / pass2_livetime

##
## Get MuonGun
##

weight_col = "cscdSBU_MuonWeight_GaisserH4a"

df_mugun = read_hdf_multiple_datasets(filename_muongun, keys=keys + [weight_col])

mugun_nfile_factor = {
    21315: 15000 / 5000,
    21316: 39995 / 5000,
    21317: 19994 / 5000,
    21318: 99975 / 5000,
    21319: 99636 / 5000,
}

nfile_factor_df = df_mugun["Run"].map(mugun_nfile_factor)
df_mugun["weights"] = df_mugun[weight_col]
df_mugun["weights"] *= 1.42 / nfile_factor_df


samples = {
    "data": dict(
        df=df_data,
        weights="weights",
        label="data",
    ),
    "mugun": dict(
        df=df_mugun, weights="weights", label="atm $\mu$ (Mugun)", color="red"
    ),
    "atm_conv": dict(
        df=df_nugen_all,
        weights="weights_conv",
        label=r"conv. $\nu$",
        color="green",
    ),
    "astro": dict(
        df=df_nugen_all,
        weights="weights_astro",
        label=r"astro $\nu$",
        color="orange",
    ),
}

for sname, sinfo in samples.items():
    df_ = sinfo["df"]
    df_["cscdSBU_MonopodFit4_cos_zenith"] = np.cos(df_["cscdSBU_MonopodFit4_zenith"])
    samples[sname]["df"] = apply_sel(df_)


variables = {
    "monopod_energy": dict(
        var="cscdSBU_MonopodFit4_energy",
        bins=np.logspace(0, 6, 51),
        label="MonopodFit4 Energy [GeV]",
        xscale="log",
    ),
    "monopod_energy_coarse": dict(
        var="cscdSBU_MonopodFit4_energy",
        # bins=np.logspace(0, 6, 10),
        bins=np.geomspace(60, 2e6, 20),
        label="MonopodFit4 Energy [GeV]",
        xscale="log",
    ),
    "monopod_energy_coarse2": dict(
        var="cscdSBU_MonopodFit4_energy",
        # bins=np.logspace(0, 6, 10),
        bins=np.geomspace(60, 2e6, 5),
        label="MonopodFit4 Energy [GeV]",
        xscale="log",
    ),
    "monopod_energy_coarse3": dict(
        var="cscdSBU_MonopodFit4_energy",
        # bins=np.logspace(0, 6, 10),
        bins=np.geomspace(60, 2e6, 7),
        label="MonopodFit4 Energy [GeV]",
        xscale="log",
    ),
    "monopod_zenith": dict(
        var="cscdSBU_MonopodFit4_zenith",
        bins=np.linspace(0, 3.25, 51),
        label="MonopodFit4 Zenith",
        xscale="linear",
    ),
    "monopod_coszenith": dict(
        var="cscdSBU_MonopodFit4_cos_zenith",
        bins=np.linspace(-1, 1, 51),
        label="MonopodFit4 Zenith",
        xscale="linear",
    ),
    "cascade_score": dict(
        var="cscdSBU_LE_bdt_cascade",
        bins=np.linspace(0, 1.1, 21),
        label="cascade BDT score",
        xscale="linear",
    ),
    "track_score": dict(
        var="cscdSBU_LE_bdt_track",
        bins=np.linspace(0, 1.1, 21),
        label="track BDT score",
        xscale="linear",
    ),
    "hybrid_score": dict(
        var="cscdSBU_LE_bdt_hybrid",
        bins=np.linspace(0, 1.1, 21),
        label="hybrid BDT score",
        xscale="linear",
    ),
    "SPEFit16FitParams_rlogl": dict(
        var="cscdSBU_LE_bdt_input_CscdL3_SPEFit16FitParams_rlogl",
        bins=np.linspace(5, 30, 51),
        # label="cascade BDT score",
        xscale="linear",
    ),
    "Qtot_HLC": dict(
        var="cscdSBU_Qtot_HLC",
        bins=np.geomspace(1, 1e5, 31),
        # label="cascade BDT score",
        xscale="log",
    ),
    "VertexRecoDist_CscdLLh": dict(
        var="cscdSBU_LE_bdt_input_cscdSBU_VertexRecoDist_CscdLLh",
        bins=np.geomspace(1e-4, 1e6, 31),
        # label="cascade BDT score",
        xscale="log",
    ),
}


sample_cuts = {"corsika": "CscdBDT==1"}


def get_hists(var, bins, samples=samples, sample_cuts=sample_cuts, **kw):
    hists = {}
    for sname, info in samples.items():
        df = info["df"]
        cut = None
        if isinstance(sample_cuts, str):
            cut = sample_cuts
        elif isinstance(sample_cuts, dict):
            cut = sample_cuts.get(sname, None)
        if cut:
            df = df.query(cut)
        weights = df[info["weights"]] if info.get("weights") else None
        h = utils.make_hist(df[var], weights=weights, bins=bins)
        hists[sname] = h
    return hists


def get_hist_and_plot(
    var, bins, plot_path=None, sample_cuts=sample_cuts, samples=samples, **var_info
):
    # var = var_info['var']
    # bins = var_info['bins']

    hists = get_hists(var, bins, samples=samples, sample_cuts=sample_cuts, **var_info)
    for sname, info in samples.items():
        df = info["df"]
        cut = None
        if isinstance(sample_cuts, str):
            cut = sample_cuts
        elif isinstance(sample_cuts, dict):
            cut = sample_cuts.get(sname, None)
        elif hasattr(sample_cuts, "__call__"):
            df = sample_cuts(df)
        if cut:
            if hasattr(cut, "__call__"):
                df = cut(df)
            elif isinstance(cut, str):
                df = df.query(cut)
            else:
                raise ValueError(f"Invalid cut type: {type(cut)}, {cut=}, ")
            # df = df.query(cut)
        weights = df[info["weights"]] if info.get("weights") else None
        h = utils.make_hist(df[var], weights=weights, bins=bins)
        hists[sname] = h

    fig, axs = utils.plt_subplots()

    ax_main = axs[0]
    hkw = dict(ax=ax_main, histtype="step", lw=1.5)
    snames = ["corsika", "mugun", "astro", "atm_conv"]

    utils.plot_hist(
        [hists[sname] for sname in snames],
        color=[samples[sname].get("color", None) for sname in snames],
        label=[samples[sname].get("label", sname) for sname in snames],
        **hkw,
    )

    snames_comp = ["corsika", "mugun"]
    utils.plot_hist_uncertainties(
        hists["corsika"],
        ax=ax_main,
        color=samples["corsika"]["color"],
        hatch=None,
        alpha=0.3,
        fill=True,
    )

    utils.plot_hist_uncertainties(
        hists["mugun"],
        ax=ax_main,
        color=samples["mugun"]["color"],
        hatch=None,
        alpha=0.3,
        fill=True,
    )

    utils.plot_error_hist(hists["data"], ax_main, color="black")

    # plot_hist(hists['data'], ax=ax_main, color='black', label='data', histtype='errorbar', lw=1.5)
    utils.plot_error_hist(hists["data"], ax_main, color="black")

    ax_r = axs[1]

    utils.plot_comparison(
        hists["data"],
        sum([hists[k] for k in ["mugun", "atm_conv", "astro"]]),
        ax=ax_r,
        color=samples["mugun"]["color"],
    )
    utils.plot_comparison(
        hists["data"],
        sum([hists[k] for k in ["corsika", "atm_conv", "astro"]]),
        ax=ax_r,
        color=samples["corsika"]["color"],
    )

    yscale = var_info.get("yscale", "log")
    xscale = var_info.get("xscale", "linear")
    ax_main.set_yscale(yscale)
    ax_main.set_xscale(xscale)
    ax_main.legend(ncol=2, loc="upper right")
    ax_main.set_ylabel("event rate [Hz]")
    y_max = max([h.values().max() for h in hists.values()])

    ax_main.set_ylim(ymax=y_max * (50 if yscale == "log" else 1.10))
    ax_main.set_ylim(ymin=1e-10)

    ax_r.set_xlabel(var_info.get("label", var))
    ax_r.set_ylabel("data/MC")
    if plot_path:
        utils.savefig(fig, plot_path)

    return dict(hists=hists, fig=fig, axs=axs)


##
## FOM Calculations
##


def fom_func_naive_sig(s_uarray, b_uarray):
    return np.sum(s_uarray) / np.sum(b_uarray).std_dev


def calc_fom(h_sig, h_bkg, per_bin=False, fom_func=fom_func_naive_sig):
    sig_values = h_sig.values()
    bkg_uncerts = np.sqrt(h_bkg.variances())
    bkg_values = h_bkg.values()
    if per_bin:
        fom_values = sig_values / bkg_uncerts
    else:
        from uncertainties import unumpy

        bkg = unumpy.uarray(bkg_values, bkg_uncerts)
        bkg = np.where(bkg == 0, np.nan, bkg)

        fom_values = []
        for ibin in range(len(bkg)):
            # fom = np.sum(sig_values[ibin:])/np.sum(bkg[ibin:]).std_dev
            fom = fom_func(sig_values[ibin:], bkg[ibin:])
            fom_values.append(fom)
            # fom_values

    return fom_values


def calc_ZA(s, b, sigma_b=0):
    if sigma_b == 0:
        return np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))
    else:
        return np.sqrt(
            2
            * (
                (s + b)
                * np.log((s + b) * (b + sigma_b**2) / (b**2 + (s + b) * sigma_b**2))
                - (b**2 / sigma_b**2)
                * np.log(1 + sigma_b**2 * s / (b * (b * sigma_b**2)))
            )
        )


def calc_Za(h_sig, h_bkg, per_bin=False):
    sig_values = h_sig.values()
    bkg_uncerts = np.sqrt(h_bkg.variances())
    bkg_values = h_bkg.values()
    if per_bin:
        zn_values = sig_values / np.sqrt(bkg_values)
    else:
        from uncertainties import unumpy

        bkg = unumpy.uarray(bkg_values, bkg_uncerts)
        bkg = np.where(bkg == 0, np.nan, bkg)

        zn_values = []
        for ibin in range(len(bkg)):
            zn = np.sum(sig_values[ibin:]) / np.sqrt(np.sum(bkg[ibin:]).std_dev)
            zn_values.append(zn)
            # fom_values

    return zn_values


def get_cpu_estimate(
    df, n_photons_col="n_photons", weight_cols=["sel_gen_weights"], time_per_photon=None
):
    n_photons = df["n_photons"]
    weights = utils.combine_weight_columns(df, weight_cols)
    return np.sum(n_photons * weights)


def get_cpu_hist(
    df, var, bins, n_photons_col="n_photons", weight_cols=["sel_gen_weights"]
):
    h = utils.make_hist(df[var], bins=bins)
    edges = h.axes[0].edges

    values = []
    print(edges)
    for edge in edges[:-1]:
        df_ = df.query(f"{var}>{edge}")
        cpu = get_cpu_estimate(
            df_, n_photons_col=n_photons_col, weight_cols=weight_cols
        )
        values.append(cpu)
        print(edge, cpu)

    return h, values


# h, values = get_cpu_hist(df, 'pred', bins=np.linspace(0,1,100))
