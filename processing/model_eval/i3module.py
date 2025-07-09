#!/usr/bin/env python3
# encoding: utf-8

"""
Example script for calculating features and passing them to a NN model
via TFModelWrapper.
"""
from __future__ import print_function, division
import timeit
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import itertools

from icecube.icetray import I3Tray
from icecube import dataclasses, icetray, dataio
from icecube import MuonGun
from icecube import ml_suite

sys.path.append("/home/navidkrad/work/i3kiss/processing/")
from ModelFactory import ModelFactory
import utils
from MCPrimaryConverter import MCPrimaryConverter
from I3ModelWrapper import MyModelWrapper

NMUONS = 10

model_name = "gen2L3_merged_combinedall_allfeatures_hyperband1"

# mf = ModelFactory.load_model(model_name=model_name)
# features = mf.features

# def model(*args, **kwargs):
#     return mf.model.predict(*args, **kwargs)

# def preprocessor(raw_data):
#     df = pd.DataFrame(itertools.chain(*raw_data))
#     df = utils.preproc(df)
#     data = df[features].to_numpy()
#     data = np.expand_dims(data, axis=0)
#     return data

# primary_converter = MCPrimaryConverter(
#     key_name="I3MCTree",
#     count_photons=False,
#     add_nus=False,
#     max_muon_multiplicity=NMUONS,
# )


# def event_feat_ext(frame):
#     row = {}
#     primary_converter.Convert(None, row, frame)
#     return row


def add_modelwrapper_to_tray(
    tray,
    model_name="gen2L3_merged_combinedall_allfeatures_hyperband1",
    n_muons=NMUONS,
    batch_size=10_000,
):
    mf = ModelFactory.load_model(model_name=model_name)
    features = mf.features

    def model(*args, **kwargs):
        return mf.model.predict(*args, **kwargs)

    def preprocessor(raw_data):
        df = pd.DataFrame(itertools.chain(*raw_data))
        df = utils.preproc(df)
        data = df[features].to_numpy()
        data = np.expand_dims(data, axis=0)
        return data

    primary_converter = MCPrimaryConverter(
        key_name="I3MCTree",
        count_photons=False,
        add_nus=False,
        max_muon_multiplicity=n_muons,
    )

    def event_feat_ext(frame):
        row = {}
        primary_converter.Convert(None, row, frame)
        return row

    tray.AddModule(
        MyModelWrapper,
        "I3ModelWrapperKeras",
        nn_model=model,
        n_inputs=len(features),
        event_feature_extractor=event_feat_ext,
        event_stream=icetray.I3Frame.DAQ,
        batch_size=batch_size,
        output_key="NNCascadeness",
        preprocessor=preprocessor,
    )
    return tray


def main():

    # test_input_file = "/data/exp/IceCube/2020/filtered/level2/0428/Run00134022/Level2_IC86.2019_data_Run00134022_Subrun00000000_00000041.i3.zst"
    test_input_file = "/data/sim/IceCube/2020/generated/CORSIKA-in-ice/20904/0000000-0000999/generated/prop.corsika.020904.000000.i3.zst"

    tray = I3Tray()
    tray.AddModule("I3Reader", "reader", Filenamelist=[test_input_file])

    add_modelwrapper_to_tray(tray, model_name=model_name, n_muons=NMUONS)
    # tray.AddModule(
    #     MyModelWrapper,
    #     "I3ModelWrapperKeras",
    #     nn_model=model,
    #     n_inputs=len(features),
    #     event_feature_extractor=event_feat_ext,
    #     event_stream=icetray.I3Frame.DAQ,
    #     # data_transformer=dummy_data_transformer,
    #     batch_size=10_000,
    #     output_key="NNCascadeness",
    #     preprocessor=preprocessor,
    # )

    tray.AddModule(
        "I3Writer",
        "EventWriter",
        filename=f"/data/user/navidkrad/test_output_i3module_batchproc_{model_name}.i3.bz2",
    )


    write_hdf = False
    if write_hdf:
        from icecube.hdfwriter import I3SimHDFWriter
        keys = [
            ("mcprimary", MCPrimaryConverter(), "PolyplopiaPrimary"),
        ]
        tray.Add(
            I3SimHDFWriter,
            keys=keys,
            output="delme.hdf5",
        )

    tray.Execute()


if __name__ == "__main__":
    main()
