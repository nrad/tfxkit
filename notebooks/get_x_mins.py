import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
sys.path.append("/cephfs/users/nrad/work/i3kiss/")
from utils.elbert_yield import ElbertYield 
import utils
from utils import run_func_in_paral
# from utils import cscd_utils
# from utils.cscd_utils import get_hist_and_plot, variables, samples, get_hists
from tqdm  import tqdm
import pickle
import argparse
import os

# output_dir_base = "/lustre/fs23/group/icecube/nrad/data/hdf/Cscd_v0.0.2_shower_mus/sim/IceCube/2020/CORSIKA-in-ice/20904/"
output_dir_base = "/lustre/fs23/group/icecube/nrad/data/hdf/Cscd_v0.0.3_HE/sim/IceCube/2020/CORSIKA-in-ice/23123/"
fnames = {
    'train': f"{output_dir_base}/test_train/train.hdf5",
    'test' :  f"{output_dir_base}/test_train/test.hdf5",
}

DEFAULT_B = [0.9999999, 0.5, 0.1, 1E-2, 0.5E-2, 1E-3, 0.5E-3, 1E-4, 1E-5, 1E-6, 1E-7]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some Bs.')
    parser.add_argument('--Bs', type=float, nargs='+', default=DEFAULT_B,
                        help='List of B values')
    args = parser.parse_args()

    Bs = args.Bs

    B_tag = f"_{Bs[0]}" if len(Bs) == 1 else ""
    fout = f"{output_dir_base}/x_mins/x_mins{B_tag}.hdf5"
    os.makedirs(os.path.dirname(fout), exist_ok=True)

    print("\nOutput file:", fout)

    dfs = []
    w_fracs = []
    for fname, w_frac in [ (fnames['test'], 0.1) , 
                           (fnames['train'],0.9)
                            ]:
        print(f'Loading {fname}')
        df = pd.read_hdf(fname)
        dfs.append(df)
        w_fracs.append(w_frac)

    df_corsika  = pd.concat(dfs)
    #df_corsika = df_corsika.sample(n=10_000)

    df_corsika['weights'] = (df_corsika['sel_flux_weights'] * 1.0/sum(w_fracs)).astype('float128')
    df_corsika['selection_weights'] = (df_corsika['selection_weights'] * 1.0/sum(w_fracs)).astype('float128')
    df_corsika['A'] = utils.get_A_from_pdg(df_corsika['pdg_encoding'])
    A = df_corsika['A']

    cos_theta = df_corsika['cos_theta']
    # x_mu = df_corsika['shower_mu1_energy']/df_corsika['energy_per_nucleon']
    targets = [ df_corsika["A"], df_corsika['energy'], df_corsika['cos_theta'],]

    x_mins = []
    n_rows = len(df_corsika)
    for target in tqdm(zip(*targets), total=n_rows, mininterval=10):
        A, primary_energy, cos_theta = target
        elbert = ElbertYield(A=A, primary_energy=primary_energy, cos_theta=cos_theta, strict=False)
        # elbert.cos_theta = cos_theta
        # elbert.cos_theta_eff = elbert.get_effective_costheta(elbert.cos_theta)
        sols = run_func_in_paral(elbert.solve_for_x_min, Bs, n_proc=1)
        sols = dict(zip(Bs, sols))
        x_mins.append(sols)


    df_x_mins = pd.DataFrame( x_mins )
    df_x_mins.rename(columns=lambda B: f'x_min_{B}', inplace=True)
    #df_corsika = pd.concat([df_corsika, df_x_mins], axis=1)
    df_x_mins.set_index(df_corsika.index, inplace=True)
    # df_corsika.loc[:, list(x_min_names)] = df_x_mins

    elbert = ElbertYield(A=df_corsika['A'], 
                            primary_energy=df_corsika['energy'],
                            cos_theta=df_corsika['cos_theta'],
                          strict=False)
    # x_mu = df['shower_mu1_energy']/df['energy_per_nucleon']
    x_mu = df_corsika['shower_mu1_energy']/(df_corsika['energy']/df_corsika['A'])
    for B in Bs:
        x_min = df_x_mins[f'x_min_{B}'].to_numpy()
        # print(f'{x_min=}')
        accpt_prob = elbert.acceptance_probability(x_mu=x_mu, x_min=x_min, B=B)
        df_x_mins[f'mu_accpt_prob_{B}'] = accpt_prob

    df_x_mins.set_index(df_corsika.index, inplace=True)
    x_min_names = df_x_mins.columns
    df_corsika.loc[:, list(x_min_names)] = df_x_mins

    #import pickle
    #pickle.dump(x_mins, open(fout, "wb"))
    print(f"Saved x_mins to {fout}")
    df_corsika[x_min_names].to_hdf(fout, key='x_mins', mode='w')


