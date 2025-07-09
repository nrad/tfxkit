import os
import defaults
from utils import unique_name
import tempfile
NMUONS = 10


def SimHDFWriter(
    infiles,
    outfile,
    sim_type,
    store_weight_info=False,
    split_muons=True,
    count_photons=True,
    event_filter=None,
    add_shower_muons=False,
    model_name=None,
    target_primary_files=None,
    acceptance_probability=0.1,
    extra_keys=[],
    temp_output=True,
):
    print("****" * 20, infiles, outfile, sim_type)

    if outfile.endswith(".hdf5") or ".hdf" in os.path.basename(outfile):
        output = outfile
    else:
        # assume a directory is given
        if not len(infiles) == 1:
            raise ValueError(
                "multiple infput files are given, cannot decide on the output filename. Give --outfile explicitly as hdf5 file."
            )

        print(" infile= %s" % infiles)
        if infiles[0].endswith(".i3.zst"):
            infile_name = os.path.basename(infiles[0]).rstrip(".i3.zst")
        else:
            infile_name = os.path.splitext(os.path.basename(infiles[0]))[0]

        output = f"{outfile}/{infile_name}.hdf5"
        os.makedirs(outfile, exist_ok=True)
        print(infile_name, output)

    if temp_output:
        output_final = output
        output = unique_name(output+".tmp", length=6)
        print(f'Writing Temporary output file: {output}')

    print(
        f"""
    ==============================
    input files:            {infiles}
    output file:            {output}
    type:                   {sim_type}
    event_filter:           {event_filter}
    store_weight_info:      {store_weight_info}
    model_name:             {model_name}
    split_muons:            {split_muons}
    count_photons:          {count_photons}
    target_primary_files:   {target_primary_files}
    acceptance_probability: {acceptance_probability}
    ==============================
    """
    )
    # assert False

    ##
    # the code begins...
    ##

    from icecube import icetray, dataclasses, dataio
    from I3Tray import I3Tray

    from icecube.tableio import I3Converter
    from icecube.tableio.I3TableWriterModule import default

    from MCPrimaryConverter import MCPrimaryConverter, SimpleFrameConverter

    tray = I3Tray()

    tray.context["I3FileStager"] = dataio.get_stagers()
    tray.Add("I3Reader", filenamelist=infiles)

    def add_primary_keys(frame, particle_name="I3MCTree", key_name="mcprimary"):
        frame[key_name] = frame[particle_name].primaries[0]

    def sort_key(
        frame,
        key="MMCTrackList",
        # sorted_key="SortedMMCTrackList",
        func=lambda x: x.GetEi(),
        reverse=True,
    ):
        from icecube import polyplopia
        # frame[sorted_key] = polyplopia.simclasses.I3MMCTrackList()
        frame.Replace(
            key,
            polyplopia.simclasses.I3MMCTrackList(
                sorted(frame[key], key=func, reverse=reverse)
            ),
        )

    from icecube.hdfwriter import I3SimHDFWriter
    from icecube.hdfwriter import I3HDFWriter

    keys = []
    hdfwriter_kwargs = dict(output=output)
    hdfwriter_args = []

    from icecube import MuonGun

    if target_primary_files:
        from processing.PrimaryParticleMatcher import PrimaryParticleMatcherModule
        tray.AddModule(PrimaryParticleMatcherModule,
                       "primary_particle_matcher",
                       TargetFile=target_primary_files,
                       AcceptanceProbability=acceptance_probability,
                       Strict=True,
                       )

    if sim_type == "detector":
        # mu_key = ('I3MCTree_preMuonProp', MuonGun.converters.MuonBundleConverter(5), "muons" )
        tray.Add(
            add_primary_keys, particle_name="I3MCTree_preMuonProp", streams=[icetray.I3Frame.DAQ]
        )
        keys = [
            ("mcprimary", MCPrimaryConverter(
                max_muon_multiplicity=0), "PolyplopiaPrimary"),
        ]

        if store_weight_info:
            keys.extend(["CorsikaWeightMap"])

    elif sim_type == "generated":

        tray.Add(add_primary_keys, particle_name="I3MCTree",
                 streams=[icetray.I3Frame.DAQ])
        # tray.Add(sort_key, sorted_key="MMCTrackList")

        tray.Add(sort_key, key="MMCTrackList", streams=[icetray.I3Frame.DAQ])

        count_photons_di = {}
        if count_photons:
            count_photons_di = dict(n_photons_per_step=1000)
            if split_muons:
                count_photons_di['key'] = "I3MCTree_sliced"
            else:
                count_photons_di['key'] = "I3MCTree"

        # count_photons = (
        #     dict(n_photons_per_step=1000, key="I3MCTree_sliced") if split_muons else {}
        # )

        keys = [
            dict(
                key="mcprimary",
                # using PolyplopiaPrimary so SimWeights can read the energy info.
                name="PolyplopiaPrimary",
                converter=MCPrimaryConverter(
                    key_name="I3MCTree",
                    count_photons=count_photons_di,
                    max_muon_multiplicity=10,
                    add_shower_muons=add_shower_muons,
                    # add_nus=True,
                ),
            ),
            "ProcessingWeight",
        ]

        # mu_key = (
        #     "I3MCTree",
        #     MuonGun.converters.MuonBundleConverter(
        #         NMUONS,  # MMCTrackList="SortedMMCTrackList"
        #     ),
        #     "muons",
        # )
        # keys.append(mu_key)
        if store_weight_info:
            keys.extend(["CorsikaWeightMap"])

        if split_muons:
            tray.AddModule(
                "I3MuonSlicer",
                "chopMuons",
                InputMCTreeName="I3MCTree",
                MMCTrackListName="MMCTrackList",
                OutputMCTreeName="I3MCTree_sliced",
            )

        if model_name:

            add_modelwrapper_to_tray(tray, model_name=model_name)
    elif sim_type == "filtered":
        # filter_names = [
        #     "CascadeFilter_13",
        #     "DeepCoreFilter_13",
        #     "EHEAlertFilterHB_15",
        #     "EHEAlertFilter_15",
        #     "EstresAlertFilter_18",
        #     "FSSCandidate_13",
        #     "FSSFilter_13",
        #     "FilterMinBias_13",
        #     "FixedRateFilter_13",
        #     "GFUFilter_17",
        #     "GRECOOnlineFilter_19",
        #     "HESEFilter_15",
        #     "HighQFilter_17",
        #     "I3DAQDecodeException",
        #     "IceActTrigFilter_18",
        #     "IceTopSTA3_13",
        #     "IceTopSTA5_13",
        #     "IceTop_InFill_STA2_17",
        #     "IceTop_InFill_STA3_13",
        #     "InIceSMT_IceTopCoincidence_13",
        #     "LowUp_13",
        #     "MESEFilter_15",
        #     "MonopoleFilter_16",
        #     "MoonFilter_13",
        #     "MuonFilter_13",
        #     "OnlineL2Filter_17",
        #     "SDST_IceTopSTA3_13",
        #     "SDST_IceTop_InFill_STA3_13",
        #     "SDST_InIceSMT_IceTopCoincidence_13",
        #     "ScintMinBias_16",
        #     "SlopFilter_13",
        #     "SunFilter_13",
        #     "VEF_13",
        # ]
        filter_names = [
            "HighQFilter_17",
            "LowUp_13",
            "MESEFilter_15",
            "MonopoleFilter_16",
            "MoonFilter_13",
            "OnlineL2Filter_17",
            "SunFilter_13",
            "VEF_13",
            "CascadeFilter_13",
            "MuonFilter_13",
            "HESEFilter_15",
            "FilterMinBias_13",
            "ScintMinBias_16",
        ]

        tray.Add(
            add_primary_keys,
            particle_name="I3MCTree_preMuonProp",
            streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics],
        )
        keys = [
            dict(
                key="mcprimary",
                converter=MCPrimaryConverter(
                    filter_names=filter_names, max_muon_multiplicity=0),
                name="PolyplopiaPrimary",
            ),
        ]
        if store_weight_info:
            keys.extend(["CorsikaWeightMap"])

    elif sim_type == "Level3":
        tray.Add(
            add_primary_keys,
            particle_name="I3MCTree_preMuonProp",
            streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics],
        )
        keys = [
            dict(
                key="mcprimary",
                converter=MCPrimaryConverter(max_muon_multiplicity=0),
                name="PolyplopiaPrimary",

            )
        ]
    elif sim_type == "CscdBDT":
        tray.Add(
            add_primary_keys,
            particle_name="I3MCTree_preMuonProp",
            streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics],
        )
        keys = defaults.default_config['cascade_keys'] + [
            dict(
                key="mcprimary",
                converter=MCPrimaryConverter(max_muon_multiplicity=0),
                name="PolyplopiaPrimary",
            )
        ]
        # key="mcprimary",
        # converter=MCPrimaryConverter(max_muon_multiplicity=0),
        # name="PolyplopiaPrimary",
#            ),
        # dict(
        #     #key="mcprimary",
        #     key="mcprimary",
        #     converter=SimpleFrameConverter(keys=["cscdSBU_Qtot_HLC"]),
        #     name="CscdSBU_info",
        # ),
#        ]
        hdfwriter_kwargs.update(dict(SubEventStreams=["InIceSplit"]))
        hdfwriter_args = [I3HDFWriter]
    else:
        raise ValueError("sim type not recognized: %s!" % sim_type)

    if extra_keys:
        keys.extend(extra_keys)

    if not hdfwriter_args:
        hdfwriter_args = [I3SimHDFWriter]
    if not 'keys' in hdfwriter_kwargs:
        hdfwriter_kwargs['keys'] = keys

    print(f'{hdfwriter_args = }')
    print(f'{hdfwriter_kwargs = }')

    if event_filter is not None:
        if not callable(event_filter):
            raise ValueError(
                f"event_filter: {event_filter} must be a callable function"
            )
        tray.Add(event_filter)

    tray.Add(*hdfwriter_args, **hdfwriter_kwargs)
    #     I3HDFWriter,
    #     output=output,
    #     **simhdf_kwargs,
    #     #streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics],
    # )

    # tray.Add(
    #     I3SimHDFWriter,
    #     keys=keys,
    #     output=output,
    #     #streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics],
    # )
    # from icecube.hdfwriter import I3HDFWriter
    # tray.Add(
    #     I3HDFWriter,
    #     keys=keys,
    #     output=output,
    #     #**simhdf_kwargs
    #     streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics],
    # )

#     from icecube.hdfwriter import I3HDFWriter
#     keys = ['TimeShift',
#  'PassedKeepSuperDSTOnly',
#  'NString_OfflinePulsesHLC_noDC',
#  'NString_OfflinePulses_noDC',
#  'CscdL2_Topo1stPulse_HLCSplitCount',
#  'NString_OfflinePulses_DCOnly',
#  'CscdL3_Cont_Tag',
#  'NCh_SRTOfflinePulses',
#  'PassedAnyFilter',
#  'NString_SRTOfflinePulses',
#  'HowManySaturDOMs',
#  'CorsikaMoonMJD',
#  'CorsikaSunMJD',
#  'CscdL2',
#  'CscdL2_Topo_HLCSplitCount',
#  'Estres_CausalQTot',
#  'Estres_Homogenized_QTot',
#  'Homogenized_QTot',
#  'NCh_OfflinePulses_DCOnly',
#  'CscdL3',
#  'NCh_OfflinePulses',
#  'NCh_OfflinePulsesHLC_noDC',
#  'NCh_OfflinePulses_noDC',
#  'NCh_OfflinePulsesHLC_DCOnly',
#  'NCh_OfflinePulsesHLC',
#  'NCh_SRTOfflinePulses_noDC',
#  'NCh_SRTOfflinePulses_DCOnly',
#  'NString_OfflinePulses',
#  'NString_OfflinePulsesHLC',
#  'NString_OfflinePulsesHLC_DCOnly',
#  'NString_SRTOfflinePulses_DCOnly',
#  'NString_SRTOfflinePulses_noDC',
#  'PoleCascadeFilter_CscdLlh',
#  'PoleCascadeFilter_LFVel',
#  'HowManySaturStrings',
#  'PoleCascadeFilter_ToiVal',
#  'SPERadius',
#  'SimTrimmer',
#  'PassedConventional']
#     # tray.Add(
#     #     I3HDFWriter,
#     #     keys=keys,
#     #     output=output,
#     #     SubEventStreams=['InIceSplit'],
#     #     #BookEverything=True,
#     # )
#     tray.Add(
#         I3HDFWriter,
#         keys=[dict(key='blah', converter=SimpleFrameConverter(keys=keys), name='online_')],
#         output=output,
#         SubEventStreams=['InIceSplit'],
#         #BookEverything=True,
#     )

    tray.Execute()
    tray.Finish()
    if temp_output:
        import shutil
        shutil.move(output, output_final)

    print("clean finish :) ")


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    infiles = [
        "/data/sim/IceCube/2020/generated/CORSIKA-in-ice/20904/0000000-0000999/generated/corsika.020904.000000.i3.zst"
    ]
    outfile = "test.hdf5"

    parser.add_argument("--infiles", nargs="+", default=infiles)
    parser.add_argument("--outfile", default=outfile)
    parser.add_argument(
        "--verify", action="store_true", help="try to read the outfile into a dataframe"
    )
    parser.add_argument(
        "--sim_type",
        default="generated",
        choices=["detector", "generated", "filtered", "Level3", "CscdBDT"],
    )

    parser.add_argument(
        "--store_weight_info",
        "-w",
        action="store_true",
    )

    parser.add_argument(
        "--split_muons",
        action="store_true",
        help="run the muon slicer",
    )
    parser.add_argument(
        "--count_photons",
        action="store_true",
        help="count photons in ice",
    )
    parser.add_argument(
        "--add_shower_muons",
        action="store_true",
        help="add shower muons to the output",
    )

    parser.add_argument("--model_name", "-m", default=None)
    parser.add_argument(
        "--target_primary_files",
        nargs="+",
        default=None,
        help="target primary file(s) for PrimaryParticleMatcherModule",
    )
    parser.add_argument(
        "--acceptance_probability",
        type=float,
        default=0.1,
        help="acceptance probability for PrimaryParticleMatcherModule",
    )

    parser.add_argument(
        "--extra_keys",
        nargs="+",
        default=[],)

    args = parser.parse_args()

    if args.model_name:
        from processing.i3module import add_modelwrapper_to_tray

    SimHDFWriter(
        args.infiles,
        args.outfile,
        args.sim_type,
        store_weight_info=args.store_weight_info,
        model_name=args.model_name,
        split_muons=args.split_muons,
        count_photons=args.count_photons,
        target_primary_files=args.target_primary_files,
        acceptance_probability=args.acceptance_probability,
        add_shower_muons=args.add_shower_muons,
        extra_keys=args.extra_keys,
    )

    if args.verify:
        from utils import get_df
        import tables
        import pandas as pd

        f = tables.open_file(args.outfile)
        members = f.root.__members__
        print("members: %s" % members)
        for member in members:
            if member.startswith("__"):
                continue
            print("  %s" % member)
            # df = pd.DataFrame( getattr(f.root, member).read() )
            # print( df.head() )
            df = get_df(args.outfile, key=member)
            print(df)
