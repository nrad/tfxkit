from utils.elbert_yield import ElbertYield
import pandas as pd
from utils import get_A_from_pdg


DEFAULT_Bs = [0.9999999, 0.5, 0.1, 1e-2, 0.5e-2, 1e-3, 0.5e-3, 1e-4, 1e-5, 1e-6, 1e-7]


def get_mu_bias_x_mins(primary_energy, A, cos_theta, Bs=DEFAULT_Bs, x_mu=None):
    """
    Get the x_min for each B in Bs.
    """
    targets = [A, primary_energy, cos_theta]

    x_mins = []
    # n_rows = len(primary_energy)
    # for target in tqdm(zip(*targets), total=n_rows, mininterval=10):
    for target in zip(*targets):
        A_, primary_energy_, cos_theta_ = target
        elbert = ElbertYield(
            A=A_, primary_energy=primary_energy_, cos_theta=cos_theta_, strict=False
        )

        sols = map(elbert.solve_for_x_min, Bs)
        sols = dict(zip(Bs, sols))
        x_mins.append(sols)

    df_x_mins = pd.DataFrame(x_mins)
    df_x_mins.rename(columns=lambda B: f"x_min_{B}", inplace=True)
    x_min_names = df_x_mins.columns

    if x_mu is not None:
        # print(f'{x_mu=}')
        elbert = ElbertYield(
            A=A, primary_energy=primary_energy, cos_theta=cos_theta, strict=False
        )
        # x_mu = df['shower_mu1_energy']
        for B in Bs:
            x_min = df_x_mins[f"x_min_{B}"].to_numpy()
            # print(f'{x_min=}')
            accpt_prob = elbert.acceptance_probability(x_mu=x_mu, x_min=x_min, B=B)
            df_x_mins[f"mu_accpt_prob_{B}"] = accpt_prob
    return df_x_mins


def get_mu_bias_from_df(
    df,
    Bs=DEFAULT_Bs,
    col_primary_energy="energy",
    col_cos_theta="cos_theta",
    col_x_mu="shower_mu1_energy",
    col_pdg="pdg_encoding",
):
    """ """
    primary_energy = df[col_primary_energy]
    cos_theta = df[col_cos_theta]
    A = get_A_from_pdg(df[col_pdg])
    x_mu = df[col_x_mu] / (primary_energy / A)
    df_x_mins = get_mu_bias_x_mins(primary_energy, A, cos_theta, Bs=Bs, x_mu=x_mu)
    df_x_mins.set_index(df.index, inplace=True)
    print(df_x_mins)
    x_min_names = list(df_x_mins)
    df.loc[:, list(x_min_names)] = df_x_mins

    return df
