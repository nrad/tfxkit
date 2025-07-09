CorsikaWeightMap_20904 = {2212.0: {'ParticleType': 2212.0,
                                   'CylinderLength': 1200.0,
                                   'CylinderRadius': 600.0,
                                   'ThetaMax': 1.570621756585442,
                                   'ThetaMin': 0.0,
                                   'PrimarySpectralIndex': -2.0,
                                   'EnergyPrimaryMin': 600.0,
                                   'EnergyPrimaryMax': 100000000.0,
                                   'OverSampling': 1.0,
                                   'NEvents': 865335.0
                                   },
                          1000020040.0: {'ParticleType': 1000020040.0,
                                         'CylinderLength': 1200.0,
                                         'CylinderRadius': 600.0,
                                         'ThetaMax': 1.570621756585442,
                                         'ThetaMin': 0.0,
                                         'PrimarySpectralIndex': -2.0,
                                         'EnergyPrimaryMin': 2400.0,
                                         'EnergyPrimaryMax': 400000000.0,
                                         'OverSampling': 1.0,
                                         'NEvents': 108167.0
                                         },
                          1000070140.0: {'ParticleType': 1000070140.0,
                                         'CylinderLength': 1200.0,
                                         'CylinderRadius': 600.0,
                                         'ThetaMax': 1.570621756585442,
                                         'ThetaMin': 0.0,
                                         'PrimarySpectralIndex': -2.0,
                                         'EnergyPrimaryMin': 8400.0,
                                         'EnergyPrimaryMax': 1400000000.0,
                                         'OverSampling': 1.0,
                                         'NEvents': 18543.0
                                         },
                          1000130270.0: {'ParticleType': 1000130270.0,
                                         'CylinderLength': 1200.0,
                                         'CylinderRadius': 600.0,
                                         'ThetaMax': 1.570621756585442,
                                         'ThetaMin': 0.0,
                                         'PrimarySpectralIndex': -2.0,
                                         'EnergyPrimaryMin': 16200.0,
                                         'EnergyPrimaryMax': 2700000000.0,
                                         'OverSampling': 1.0,
                                         'NEvents': 6410.0
                                         },
                          1000260560.0: {'ParticleType': 1000260560.0,
                                         'CylinderLength': 1200.0,
                                         'CylinderRadius': 600.0,
                                         'ThetaMax': 1.570621756585442,
                                         'ThetaMin': 0.0,
                                         'PrimarySpectralIndex': -2.0,
                                         'EnergyPrimaryMin': 33600.0,
                                         'EnergyPrimaryMax': 5600000000.0,
                                         'OverSampling': 1.0,
                                         'NEvents': 1545.0
                                         } 
                                         }

CorsikaWeightMap_23123 = {2212.0: {'ParticleType': 2212.0,
                'CylinderLength': 1600.0,
                'CylinderRadius': 800.0,
                'ThetaMax': 1.570621756585442,
                'ThetaMin': 0.0,
                'PrimarySpectralIndex': -2.0,
                'EnergyPrimaryMin': 100000000.0,
                'EnergyPrimaryMax': 99999997952.0,
                'OverSampling': 1.0,
                'NEvents': 476.0
                },
                1000020040.0: {'ParticleType': 1000020040.0,
                'CylinderLength': 1600.0,
                'CylinderRadius': 800.0,
                'ThetaMax': 1.570621756585442,
                'ThetaMin': 0.0,
                'PrimarySpectralIndex': -2.0,
                'EnergyPrimaryMin': 100000000.0,
                'EnergyPrimaryMax': 99999997952.0,
                'OverSampling': 1.0,
                'NEvents': 238.0
                },
                1000070140.0: {'ParticleType': 1000070140.0,
                'CylinderLength': 1600.0,
                'CylinderRadius': 800.0,
                'ThetaMax': 1.570621756585442,
                'ThetaMin': 0.0,
                'PrimarySpectralIndex': -2.0,
                'EnergyPrimaryMin': 100000000.0,
                'EnergyPrimaryMax': 99999997952.0,
                'OverSampling': 1.0,
                'NEvents': 143.0
                },
                1000130270.0: {'ParticleType': 1000130270.0,
                'CylinderLength': 1600.0,
                'CylinderRadius': 800.0,
                'ThetaMax': 1.570621756585442,
                'ThetaMin': 0.0,
                'PrimarySpectralIndex': -2.0,
                'EnergyPrimaryMin': 100000000.0,
                'EnergyPrimaryMax': 99999997952.0,
                'OverSampling': 1.0,
                # 'NEvents': 95.0
                },
                1000260560.0: {'ParticleType': 1000260560.0,
                'CylinderLength': 1600.0,
                'CylinderRadius': 800.0,
                'ThetaMax': 1.570621756585442,
                'ThetaMin': 0.0,
                'PrimarySpectralIndex': -2.0,
                'EnergyPrimaryMin': 100000000.0,
                'EnergyPrimaryMax': 99999997952.0,
                'OverSampling': 1.0,
                'NEvents': 48.0
                }}
# CorsikaWeightMap_23123 = {2212.0: {'ParticleType': 2212.0,


wmap_cols = ['ParticleType', 'CylinderLength', 'CylinderRadius', 'ThetaMax', 'ThetaMin',
             'PrimarySpectralIndex', 'EnergyPrimaryMin', 'EnergyPrimaryMax', 'OverSampling', 'NEvents'
             ]


CorsikaWeightMaps = {20904: CorsikaWeightMap_20904}


def get_weight_map(hdf_file):
    from utils import get_df
    wmap = get_df(hdf_file, key="CorsikaWeightMap")
    weight_map = dict(
        [(k[0], dict(zip(wmap_cols, k))) for k in list(wmap.groupby(wmap_cols).groups)])
    return weight_map

# merged = "/data/user/navidkrad/hdf/cascades_v0.5_L3HESEFilter/sim/IceCube/2020/CORSIKA-in-ice/20904/0070000-0070999/MergedRuns_preproc_0070000-0070999.hdf5"
# df = pd.read_hdf(merged)


# pdg_map_inverse = {v:k for k,v in utils.pdg_map.items()}
pdg_map_inverse = {0: 2212, 1: 1000020040,
                   2: 1000070140, 3: 1000130270, 4: 1000260560}

primaries = [2212, 1000020040, 1000070140, 1000130270, 1000260560]


def get_corsika_weight_columns(df, dataset=20904, pdg_column='pdg_encoding'):
    import pandas as pd
    weight_map = CorsikaWeightMaps[dataset]

    # df = pd.DataFrame(index=df.index, columns=[pdg_column])
    df = df[[pdg_column]] 
    #for pdg_indx, pdg in pdg_map_inverse.items():
    for pdg in primaries:
        mask = df[pdg_column] == pdg
        df.loc[mask, 'type'] = pdg
        # print(pdg, mask.sum(), df[pdg_column] ) 
        for col in wmap_cols:
            # print(pdg, col, weight_map[pdg][col])
            df.loc[mask, col] = weight_map[pdg][col]
    return df




def get_flux_weights_from_df(df, nfiles=20, flux=None):
    import simweights
    weighter = simweights.CorsikaWeighter({'CorsikaWeightMap': df[wmap_cols], 'PolyplopiaPrimary': df[[
                                          'energy', 'zenith', 'type']]}, nfiles=nfiles)
    if flux is None:
        flux = simweights.GaisserH4a()
    w_flux = weighter.get_weights(flux)
    print(weighter.tostring(flux))
    print(livetime(w_flux))
    return w_flux


