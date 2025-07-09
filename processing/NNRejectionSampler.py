#!/usr/bin/env python3
# encoding: utf-8

"""
Script for applying a rejection sampling based on a pretrained NN model.
The NN prediction is used as a probability for accepting or rejecting the shower.
The prediction is also stored in the frame to be used as the inversed of the sampling weight.

Functions:
- make_rejection_sampler: Creates a rejection sampler function based on specified parameters.
- add_rejection_sampler: Adds a rejection sampler module to the processing pipeline.
- add_model_wrapper: Adds a model wrapper module to the processing pipeline.
- main: Executes the main processing pipeline.

"""

from __future__ import print_function, division
import timeit
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import sys
import itertools
import os


from icecube.icetray import I3Tray
from icecube import icetray, phys_services

# from icecube import MuonGun
# from icecube import ml_suite


sys.path.append("/home/navidkrad/work/i3kiss/processing/")
# from ModelFactory import ModelFactory
import utils
from MCPrimaryConverter import MCPrimaryConverter
from I3ModelWrapper import MyModelWrapper

# NMUONS = 10
# model_name = "gen2L3_merged_combinedall_allfeatures_hyperband1"

DEFAULT_INPUT = "/data/sim/IceCube/2020/generated/CORSIKA-in-ice/20904/0000000-0000999/generated/prop.corsika.020904.000000.i3.zst"
DEFAULT_NMUONS = 10
DEFAULT_OUTPUT = "corsika_sampled.i3.zst"
DEFAULT_OUTPUT_KEY = "NNCascadeness"
DEFAULT_PRED_KEY = "prediction_0000"
DEFAULT_MIN_PRED = 1e-5

DEFAULT_MODEL_NAME = "gen2L3_merged_combinedall_allfeatures_hyperband1"
DEFAULT_BATCH_SIZE = 10_000


def make_rejection_sampler(
    output_key=DEFAULT_OUTPUT_KEY,
    pred_key=DEFAULT_PRED_KEY,
    minimum_threshold=DEFAULT_MIN_PRED,
    rng=None,
):
    def rejection_sampler(frame, **kwargs):
        pred = frame[output_key][pred_key]
        if rng is None:
            rand = np.random.random()
        else:
            rand = rng.uniform(0, 1)
        if minimum_threshold:
            pred = max(pred, minimum_threshold)
        return pred > rand

    return rejection_sampler


def add_rejection_sampler(
    tray,
    name="EventFilter",
    output_key=DEFAULT_OUTPUT,
    pred_key=DEFAULT_PRED_KEY,
    streams=[icetray.I3Frame.DAQ],
    rng=None,
    minimum_threshold=DEFAULT_MIN_PRED,
):

    rejection_sampler = make_rejection_sampler(
        output_key=output_key,
        pred_key=pred_key,
        rng=rng,
        minimum_threshold=minimum_threshold,
    )
    tray.Add(rejection_sampler, name, streams=streams)


def add_model_wrapper(
    tray,
    model_name=DEFAULT_MODEL_NAME,
    n_muons=DEFAULT_NMUONS,
    batch_size=DEFAULT_BATCH_SIZE,
    output_key=DEFAULT_OUTPUT_KEY,
):
    # model_dir_base = "/net/cvmfs_users/navidkrad/NNRejSampler/models/"
    # model_dir_base = "/data/user/navidkrad/i3kiss/models/"
    model_dir_base = (
        "/cvmfs/icecube.opensciencegrid.org/users/navidkrad/NNRejSampler/models/"
    )

    version_tag = "cascades_v0.2_L3"
    model_dir = os.path.join(model_dir_base, version_tag, model_name)

    try:
        import ModelFactory

        mf = ModelFactory.ModelFactory.load_model(
            model_dir_base="",
            version_tag="",
            model_name=model_dir,
        )
        print(f"------------------ \n{mf.model_dir}\n{mf.model_name}\n{mf.version_tag}")
        features = mf.features
        keras_model = mf.model

    except ImportError:
        import yaml

        # model_dir = os.path.join(model_dir_base, version_tag, model_name)
        model_path = os.path.join(model_dir, "model.keras")
        config_path = os.path.join(model_dir, "config.yml")

        model_config = yaml.safe_load(open(config_path, "r"))
        features = model_config["features"]
        keras_model = keras.models.load_model(model_path)
        keras_model.load_weights(os.path.join(model_dir, "weights.h5"))

    def nn_model(*args, **kwargs):
        return keras_model.predict(*args, **kwargs)

    def preprocessor(raw_data):
        print(f'{raw_data = }')
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
        nn_model=nn_model,
        n_inputs=len(features),
        event_feature_extractor=event_feat_ext,
        event_stream=icetray.I3Frame.DAQ,
        batch_size=batch_size,
        output_key=output_key,
        preprocessor=preprocessor,
    )
    return tray


def main(
    input_file,
    output_file,
    model_name=DEFAULT_MODEL_NAME,
    output_key=DEFAULT_OUTPUT_KEY,
    minimum_threshold=DEFAULT_MIN_PRED,
    pred_key=DEFAULT_PRED_KEY,
    n_muons=DEFAULT_NMUONS,
    rng=None,
    batch_size=DEFAULT_BATCH_SIZE,
):

    tray = I3Tray()
    tray.AddModule("I3Reader", "reader", Filenamelist=[input_file])

    add_model_wrapper(
        tray,
        model_name=model_name,
        n_muons=n_muons,
        batch_size=batch_size,
    )
    add_rejection_sampler(
        tray,
        output_key=output_key,
        pred_key=pred_key,
        rng=rng,
        minimum_threshold=minimum_threshold,
    )

    tray.AddModule(
        "I3Writer",
        "EventWriter",
        filename=output_file,
    )

    write_hdf = False
    if write_hdf:
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Script for applying a rejection sampling based on a pretrained NN model."
    )
    parser.add_argument(
        "--input_file", type=str, help="Path to the input file", default=DEFAULT_INPUT
    )

    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to the output file",
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--output_key",
        type=str,
        default=DEFAULT_OUTPUT_KEY,
        help="Key for the output in the frame",
    )
    parser.add_argument(
        "--minimum_threshold",
        type=float,
        default=DEFAULT_MIN_PRED,
        help="Minimum threshold for accepting the event",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Name of the model to use for rejection sampling",
    )
    parser.add_argument(
        "--pred_key",
        type=str,
        default=DEFAULT_PRED_KEY,
        help="Key for the prediction in the output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the random number generator",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Seed for the random number generator",
    )

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    output_key = args.output_key
    minimum_threshold = args.minimum_threshold
    model_name = args.model_name
    pred_key = args.pred_key
    seed = args.seed
    batch_size = args.batch_size

    rng = phys_services.I3GSLRandomService(seed) if seed else None

    main(
        input_file,
        output_file,
        output_key=output_key,
        minimum_threshold=minimum_threshold,
        model_name=model_name,
        pred_key=pred_key,
        rng=rng,
        batch_size=batch_size,
    )
