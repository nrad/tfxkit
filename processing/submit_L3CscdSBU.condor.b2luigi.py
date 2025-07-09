from time import sleep
import os
import subprocess
import sys
import glob
import re

from pprint import pprint


import itertools
# from submit.helpers import split_list_in_chunks, natural_sort
from utils import split_list_in_chunks, natural_sort
import utils
# from defaults import DEFAULT_FILTER_NAMES
import submit_utils

ENVSHEL = "/home/navidkrad/work/i3kiss/run_i3_command.sh"


try:
    import b2luigi
# except ImportError:
except ModuleNotFoundError:
    # need to point to the b2luigi location
    print("failed to load b2luigi... will append the sys.path")
    sys.path.append("/home/navidkrad/.local/lib/python3.11/site-packages")
    import b2luigi

# load badfiles
bad_files = [
    f.strip()
    for f in open("/home/navidkrad/work/i3kiss/processing/badfiles", "r").readlines()
    if not f.startswith("#")
]


input_infos = {
    "generated": {
        "directory": "/data/sim/IceCube/2020/generated/CORSIKA-in-ice/{dataset}/{run_range}/generated/",
        "filename": "prop.corsika.{dataset:06}.{run}.i3.zst",
        "store_weight_info": True,
    },
    "detector": {
        "directory": "/data/sim/IceCube/2020/generated/CORSIKA-in-ice/{dataset}/{run_range}/detector/",
        "filename": "IC86.2020_corsika.{dataset:06}.{run}.i3.zst",
    },
    "filtered": {
        "directory": "/data/sim/IceCube/2020/filtered/level2/CORSIKA-in-ice/{dataset}/{run_range}/",
        "filename": "Level2_IC86.2020_corsika.{dataset:06}.{run}.i3.zst",
    },
    "Level3": {
        "directory": "/data/user/navidkrad/data/sim/IceCube/2020/filtered/level3/CORSIKA-in-ice/{dataset}/{run_range}/",
        "filename": "Level3_IC86.2020_corsika.{dataset:06}.{run}.i3.zst",
    },
    # "CscdPreBDT": {
    #     "directory": "/data/user/navidkrad/data/sim/IceCube/2020/filtered/cscdSBU_prebdt/CORSIKA-in-ice/{dataset}/{run_range}/",
    #     "filename": "CscdSBU_PreBDT_IC86.2020_corsika.{dataset:06}.{run}.i3.zst",
    # },
    "CscdBDT": {
        "directory": "/data/user/navidkrad/data/sim/IceCube/2020/filtered/cscdSBU_bdt/CORSIKA-in-ice/{dataset}/{run_range}/",
        "filename": "CscdSBU_BDT_IC86.2020_corsika.{dataset:06}.{run}.i3.zst",
    },


}

dataset = 20904
output_sub_dir = "/sim/IceCube/2020/CORSIKA-in-ice/{dataset}/{run_range}"


sim_types = list(input_infos)

BATCH = True
TEST = False

version_tag = "cascades_v0.1_L3"
version_tag = "cascades_v0.2_L3"  # sel frac
version_tag = "cascades_v0.5_L3HESEFilter"  # sel frac
version_tag = "cascades_v0.6_CscdSBU_PreBDT"  # sel frac
version_tag = "cascades_v0.6_CscdSBU_BDT"  # sel frac

version_tag = "Cscd_v0.0.1_mupos"  # extra mu info and ProcessingWeight
version_tag = "Cscd_v0.0.2_mupos"  # extra mu info and ProcessingWeight smaller sel_frac for gen and det
version_tag = "Cscd_v0.0.2_shower_mus"  # extra mu info and ProcessingWeight smaller sel_frac for gen and det

NFILES = 250  # process 20 files together

NFILES_TOTAL = 99880 # !!!!! **** !!!!! SUPER HARD CODED WARNING !!!! ***** !!!!  

if TEST:
    # NCHUNKS = 10
    # NFILES = 10
    # NRUNS = 1
    # version_tag = version_tag + "_TEST"
    NCHUNKS = 10
    NFILES = 3
    NRUNS = 2
    version_tag = version_tag + "_TEST"


else:
    NCHUNKS = None
    NRUNS = 50  # number of run directories
    NRUNS = 20  # number of run directories
    NRUNS = (0, 100)  # number of run directories


output_dir_base = f"/data/user/navidkrad/hdf/{version_tag}/{output_sub_dir}"

print(f"{output_dir_base = }")

# @lru_cache()
# def gather_files(input_infos, dataset=dataset, n_runs=NRUNS, match=True):
#     print("gather files:")
#     pprint(input_infos)
#     input_dataset_generated = input_infos["generated"]["directory"]
#     generated_dirs = natural_sort(
#         glob.glob(input_dataset_generated.format(dataset=dataset, run_range="*"))
#     )
#     run_ranges = [re.search(r"(\d+-\d+)", path).group() for path in generated_dirs]

#     if isinstance(n_runs, int):
#         run_ranges = run_ranges[:n_runs]
#     elif isinstance(n_runs, (list, tuple)):
#         assert len(n_runs) == 2
#         run_ranges = run_ranges[n_runs[0] : n_runs[1]]

#     input_files = {}
#     print(run_ranges)
#     for run_range in run_ranges:
#         input_files[run_range] = {}
#         for sim_type in sim_types:
#             input_info = input_infos[sim_type]
#             input_path = os.path.join(input_info["directory"], input_info["filename"])
#             input_path = input_path.format(
#                 run_range=run_range, dataset=dataset, run="*"
#             )

#             run_range_files = natural_sort(glob.glob(input_path))
#             run_pattern = input_infos[sim_type]["filename"].format(
#                 dataset=dataset, run=r"(\d+)"
#             )

#             for filename in run_range_files:
#                 run = re.search(run_pattern, filename).group(
#                     1
#                 )  # if this fails we have a problem
#                 if run not in input_files[run_range]:
#                     input_files[run_range][run] = {}
#                 input_files[run_range][run][sim_type] = filename
#     if match:
#         for run_range in list(input_files):
#             for run in list(input_files[run_range]):
#                 if not len(input_files[run_range][run]) == len(sim_types):
#                     print(
#                         f"For {run_range=}, {run= } found only {list(input_files[run_range][run])}. Will Drop it!"
#                     )
#                     input_files[run_range].pop(run)
#     return input_files


# def get_input_output_split_by_chunk():
#     print("getting args")
#     input_files_all = gather_files(
#         input_infos, dataset=dataset, n_runs=NRUNS, match=True
#     )
#     args = {}
#     for run_range in input_files_all:
#         run_chunks = list(
#             split_list_in_chunks(list(input_files_all[run_range]), NFILES)
#         )
#         args[run_range] = {}
#         n_chunks = len(run_chunks)
#         for ichunk, run_chunk in enumerate(run_chunks):
#             input_files_chunk = [input_files_all[run_range][run] for run in run_chunk]
#             input_files_gen = [fs["generated"] for fs in input_files_chunk]
#             input_files_det = [fs["detector"] for fs in input_files_chunk]
#             input_files_filt = [fs["filtered"] for fs in input_files_chunk]
#             input_files_l3 = [fs["Level3"] for fs in input_files_chunk]

#             sub_dir = [x.split("detector")[0] for x in input_files_det]
#             assert len(set(sub_dir)) == 1, (
#                 "Failed to obtain a unique sub-dir from input_files. Got these sub-dirs: %s"
#                 % sub_dir
#             )
#             sub_dir = sub_dir[0].lstrip("/data/")

#             output_filename = f"{{sim_type}}/{{sim_type}}_{run_range}_{ichunk+1:02}_of_{n_chunks:02}.hdf5"
#             output_pattern = output_dir_base.format(
#                 dataset=dataset, run_range=run_range
#             )
#             output_pattern = os.path.join(output_pattern, output_filename)
#             args[run_range][ichunk] = dict(
#                 input_files_gen=input_files_gen,
#                 input_files_det=input_files_det,
#                 input_files_filt=input_files_filt,
#                 input_files_l3=input_files_l3,
#                 output_pattern=output_pattern,
#             )

#             if NCHUNKS and ichunk >= NCHUNKS:
#                 print(f"TEST MODE: skipping chunks >= {NCHUNKS}")
#                 break
#     return args


reload_input_output = False

if reload_input_output:
    input_output = submit_utils.get_input_output_split_by_chunk(
        input_infos=input_infos,
        dataset=20904,
        nruns=None,
        nchunks=None,
        nfiles=NFILES,
        #output_dir_base="/data/user/navidkrad/data/sim/IceCube/2020/filtered/level3/CORSIKA-in-ice/{dataset}/{run_range}/",
        output_dir_base=output_dir_base,
    )
    import pickle
    pickle.dump(input_output, open(f"input_output_{dataset}.pkl", "wb"))
else:
    import pickle
    input_output = pickle.load(open("input_output.pkl", "rb"))

print(f"{input_output.keys() = }")


class SimHDFWriterWrapper(b2luigi.Task):
    """

        Wrapper for the SimHDFWriter script

    """
    sim_type = b2luigi.Parameter(default="detector")
    input_files = b2luigi.ListParameter(hashed=True)
    output_file = b2luigi.Parameter(hashed=True)
    acceptance_probability = b2luigi.FloatParameter(default=0.0)
    target_primary_files = b2luigi.ListParameter(default=[], hashed=True)

    script = "/home/navidkrad/work/i3kiss/processing/SimHDFWriter.py"

    def output(self):
        yield self.add_to_output(self.output_file)

    @b2luigi.on_temporary_files
    #def run(self):
    def run(self):
        output_file_name = self.get_output_file_name(self.output_file)

        if "-luigi-tmp" in output_file_name:
            original_name = output_file_name.split("-luigi-tmp")[0]
        else:
            original_name = output_file_name

        print(
            f""" SIMHDFWriterWrapper DEBUG
              {output_file_name = }
              {os.path.isfile(output_file_name) = }
              {original_name  = }   
              """
        )

        if os.path.isfile(original_name):
            print(
                "   WARNING!! THE OUTPUT EXISTS BUT THIS TASK IS RUNNING FOR SOMEREASON!!"
            )
            return  # this task is already done

        self.command = "python {script} --infiles {infiles} --sim_type {sim_type} {extra_args}"
        # extra_args = '--store_weight_info' if self.sim_type=='detector' else ''
        # extra_args = "" if self.sim_type == "generated" else "--store_weight_info"
        extra_args = "--store_weight_info"
        infiles = self.input_files
        if self.target_primary_files:
            self.command += " --target_primary_files " + " ".join(self.target_primary_files)
            self.command += " --acceptance_probability %s" % self.acceptance_probability
        if self.sim_type == "generated":
            self.command += " --count_photons"
            self.command += " --split_muons"
            self.command += " --add_shower_muons"
        
        self.command += " --outfile {outfile}"



        found_badfile = False
        for f in infiles:
            if f in bad_files:
                found_badfile = True
                print("FOUND BAD FILE: %s" % f)
        if found_badfile:
            infiles = [f for f in infiles if f not in bad_files]

        self.command = self.command.format(
            script=self.script,
            infiles=" ".join(infiles),
            outfile=output_file_name,
            sim_type=self.sim_type,
            extra_args=extra_args,
        )
        print(ENVSHEL + " " + self.command)
        subprocess.check_call(ENVSHEL + " " + self.command, shell=True)


class HDFCombinerWrapper(b2luigi.Task):
    # batch_system = "local"
    htcondor_settings = {
        "request_memory": "8 GB"
    }


    input_files_gen = b2luigi.ListParameter(hashed=True)
    input_files_det = b2luigi.ListParameter(hashed=True)
    input_files_filt = b2luigi.ListParameter(hashed=True)
    input_files_l3 = b2luigi.ListParameter(hashed=True)
    #input_files_cscdprebdt = b2luigi.ListParameter(hashed=True)
    input_files_cscdbdt = b2luigi.ListParameter(hashed=True)
    output_pattern = b2luigi.Parameter(hashed=True)
    preproc = b2luigi.BoolParameter(default=False)
    apply_selection_fractions = b2luigi.BoolParameter(default=True)
    # filter_names = b2luigi.ListParameter(default=DEFAULT_FILTER_NAMES)

    selection_fractions = {
        "generated":  0.0010,
        "triggered":  0.0100,
        "filtered":   1.0000,
        "Level3":     1.0000,
        #"CscdPreBDT": 1.0000,
        "CscdBDT": 1.0000,
    }

    script = "/home/navidkrad/work/i3kiss/processing/CombineGenDet.py"
    key = "PolyplopiaPrimary"
    columns = [
        "x",
        "y",
        "z",
        "length",
        "pdg_encoding",
        "energy",
        "interaction_height",
        "zenith",
        "azimuth",
        "theta",
        "n_photons",
        "majorID",
        # "n_mu",
        # "nu1_energy",
        # "nu2_energy",
    ]
    index = ["energy", "minorID", "z"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hdf5_file_gen = self.output_pattern.format(sim_type="generated")
        self.hdf5_file_det = self.output_pattern.format(sim_type="detector")
        self.hdf5_file_filt = self.output_pattern.format(sim_type="filtered")
        self.hdf5_file_l3 = self.output_pattern.format(sim_type="Level3")
        #self.hdf5_file_cscdprebdt = self.output_pattern.format(sim_type="CscdPreBDT")
        self.hdf5_file_cscdbdt = self.output_pattern.format(sim_type="CscdBDT")
        # self.output_file_comb = self.output_pattern.format(sim_type="combined")
        sim_type = "combined" + ("_preproc" if self.preproc else "")

        if self.apply_selection_fractions:
            self.sel_frac_arg = "--selection_fractions " + " ".join(
                "%s %s" % (k, v) for k, v in self.selection_fractions.items()
            )
            self.sel_frac_arg += " --selection_weight_key selection_weights_partial"
            #
            sim_type += "_presel"
        else:
            self.sel_frac_arg = ""

        self.output_file_comb = self.output_pattern.format(sim_type=sim_type)

    def requires(self):
        yield SimHDFWriterWrapper(
            #input_files=self.input_files_cscdprebdt,
            input_files=self.input_files_cscdbdt,
            #output_file=self.hdf5_file_cscdprebdt,
            output_file=self.hdf5_file_cscdbdt,
            #sim_type="CscdPreBDT",
            sim_type="CscdBDT",
        )
        yield SimHDFWriterWrapper(
            input_files=self.input_files_l3,
            output_file=self.hdf5_file_l3,
            sim_type="Level3",
        )
        yield SimHDFWriterWrapper(
            input_files=self.input_files_filt,
            output_file=self.hdf5_file_filt,
            sim_type="filtered",
        )
        yield SimHDFWriterWrapper(
            input_files=self.input_files_det,
            output_file=self.hdf5_file_det,
            sim_type="detector",
        )
        yield SimHDFWriterWrapper(
            input_files=self.input_files_gen,
            output_file=self.hdf5_file_gen,
            sim_type="generated",
            target_primary_files=self.input_files_l3,
            acceptance_probability=0.01,
            # add_shower_muons=True, 
        )

    def output(self):
        yield self.add_to_output(self.output_file_comb)

    @b2luigi.on_temporary_files
    def run(self):

        output_file_name = self.get_output_file_name(
            self.output_file_comb
        )  # handles temp file creation
        self.command = f"python {self.script}"

        command_args = [
            f"--input_file_gen  {self.hdf5_file_gen}",
            f"--input_file_det  {self.hdf5_file_det}",
            f"--input_file_filt {self.hdf5_file_filt}",
            f"--input_file_l3   {self.hdf5_file_l3}",
            f"--input_file_cscdbdt   {self.hdf5_file_cscdbdt}",
            f"--output_file    {output_file_name}",
            f'--columns {" ".join(self.columns)}',
            f'--index   {" ".join(self.index)}',
            f"--key     {self.key}",
            f"--gen_weights_nfiles {NFILES_TOTAL}",
            f"--keep_filters",
            f"{self.sel_frac_arg}",
        ]

        if self.preproc:
            command_args.append("--preproc")

        self.command = self.command + " " + " ".join(command_args)
        print(self.command)
        subprocess.check_call(ENVSHEL + " " + self.command, shell=True)


class RunMergerTask(b2luigi.Task):
    run_range = b2luigi.Parameter()
    filter_name = b2luigi.Parameter(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.filter_tag = f"_{self.filter_name}" if self.filter_name else ""
        output_file = os.path.join(
            output_dir_base, "MergedRuns_preproc%s_{run_range}.hdf5"%self.filter_tag
        ).format(
            dataset=dataset,
            run_range=self.run_range,
        )
        self.output_file = output_file

    def output(self):
        return self.add_to_output(self.output_file)

    def requires(self):
        run_range = self.run_range
        for ichunk in input_output[run_range]:
            yield HDFCombinerWrapper(**input_output[run_range][ichunk], preproc=True)

    @b2luigi.on_temporary_files
    def run(self):
        from utils import combine_and_preproc

        input_files = itertools.chain(*list(self.get_input_file_names().values()))
        input_files = list(input_files)

        print(f"RunMergerTask: {len(input_files) = }")
        postselection = None 
        if self.filter_name: 
            postselection = f'(({self.filter_name}==1) & (filtered==1))'
        combine_and_preproc(
            input_files,
            output=self.output_file,
            balance_by=None,
            preselection=None,
            do_preproc=False,
            postselection=postselection,
        )




class TestTrainMakerTask(b2luigi.Task):

    #inputs = b2luigi.ListParameter()
    tag = b2luigi.Parameter(default="")
    batch_system = "local"
    # htcondor_settings = {
    #     "request_memory": "32 GB"
    # }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tag = f"_{self.tag}" if self.tag else ""
        output_dir = os.path.join(
            output_dir_base, "test_train%s"%self.tag, 
        ).format(
            dataset=dataset,
            run_range="",
        )
        self.output_file_test = os.path.join(output_dir, "test.hdf5")
        self.output_file_train = os.path.join(output_dir, "train.hdf5")
        # self.output_file = output_file
        input_dict = self.get_input_file_names()
        self.input_files = natural_sort(list(itertools.chain(*input_dict.values())))


    def output(self):
        yield self.add_to_output(self.output_file_test)
        yield self.add_to_output(self.output_file_train)
    

    @b2luigi.on_temporary_files
    def run(self):
        from utils import combine_and_preproc, split_file_list
        from utils.tf_utils import add_balanced_weights

        # assert False

        def add_columns(df):
            #for label in ["Level3", "CscdPreBDT"]:

            # df['selection_weights_partial'] = df[s'selection_weights']
            df['selection_weights'] = df['selection_weights_partial'] * df['ProcessingWeight']

            for label in ["Level3", "CscdBDT"]:
                add_balanced_weights(df, 
                                     label=label, 
                                     weight_col="flux_weights",
                                     balanced_weight_col="balanced_weights_%s"%label,
                                     )

                add_balanced_weights(df, 
                                     label=label, 
                                     weight_col=["flux_weights", "ProcessingWeight"],
                                     balanced_weight_col="balanced_proc_weights_%s"%label,
                                     )

                add_balanced_weights(df, 
                                     label=label, 
                                     weight_col="flux_weights",
                                     balanced_weight_col="balanced_weights_squared_%s"%label,
                                     weight_func=lambda x: x**2,
                                     )

                add_balanced_weights(df, 
                                     label=label, 
                                     weight_col=["flux_weights", "ProcessingWeight"],
                                     balanced_weight_col="balanced_proc_weights_squared_%s"%label,
                                     weight_func=lambda x: x**2,
                                     )

            df['balanced_weights'] = df['balanced_weights_CscdBDT']
            df['balanced_weights_squared'] = df['balanced_weights_squared_CscdBDT']

            df['truth'] = df['CscdBDT']
            df["sel_flux_weights"] = df["flux_weights"] * df["selection_weights"]

            return df


        print(f'{self.input_files = }')

        file_list_train, _, file_list_test = split_file_list(self.input_files, val_frac=0, test_frac=0.1)
        print(f"{len(self.input_files) = }")
        print(f"{len(file_list_train) = }")
        print(f"{len(file_list_test) = }")
        assert( len(file_list_train) + len(file_list_test) == len(self.input_files) )
        for infiles, outfile in [ (file_list_train, self.output_file_train), 
                                 (file_list_test,  self.output_file_test) ]:
            print(f"infiles: {infiles}")
            print(f"outfile: {outfile}")
            combine_and_preproc(
               infiles,
               output=outfile,
               balance_by=None,
               preselection=None,
               do_preproc=add_columns,
               #postselection=postselection,
            )


    def requires(self):
        for run_range in input_output:
            yield RunMergerTask(run_range=run_range, filter_name=None)

    # run_ranges = list(input_output)

    # combined_runs = list(utils.split_list_in_chunks(run_ranges, 10))
    # for run_ranges in combined_runs:
    #     run_starts_ends = [r.split("-") for r in run_ranges]
    #     run_starts_ends_flat = list(
    #         itertools.chain(*[(int(r[0]), int(r[1])) for r in run_starts_ends])
    #     )

    # def output(self):
    #     pass


class Wrapper(b2luigi.Task):
    batch_system = "local"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def requires(self):
        yield TestTrainMakerTask(tag="")
        # for run_range in input_output:
        #     yield RunMergerTask(run_range=run_range)


    def output(self):
        # yield self.add_to_output(f"{version_tag}/test.txt")
        yield self.add_to_output(f"{version_tag}/test.txt")

    def run(self):
        with open(self.get_output_file_name(f"{version_tag}/test.txt"), "w") as f:
            # f.write(f"{self.output_det}\n{self.output_gen}")
            f.write("done!")


if __name__ == "__main__":
    # Choose htcondor as our batch system
    b2luigi.set_setting("batch_system", "htcondor")
    b2luigi.set_setting("log_dir", f"./logs/{version_tag}/")
    #b2luigi.set_setting("log_dir", f"/scratch/navidkrad/condor/b2luigi/logs/{version_tag}/")
    # b2luigi.set_setting("log_dir", f"/data/user/navidkrad/junk/b2luigi/logs/{version_tag}/")
    # b2luigi.set_setting("log_dir", f"/scratch/navidkrad/junk/b2luigi/logs/{version_tag}/")
    b2luigi.set_setting(
        "result_dir", f"/data/user/navidkrad/junk/b2luigi/results/{version_tag}/"
    )
    b2luigi.set_setting(
        "htcondor_settings",
        {
            #'Requirements  = (OpSysAndVer == "CentOS7")'
            # "should_transfer_files": "YES",
            # "MaxJobRetirementTime": 10000,
            # "+RequestRuntime": 90*3600, # 60 hrs!
            # "Request_Memory": "4GB",
            "Request_Memory": "4GB",
            # "When_to_transfer_output" : "ON_EXIT",
        },
    )

    # Setup the correct environment on the workers
    b2luigi.set_setting(
        # "env_script", "i3setup.sh"
        "env_script",
        "/home/navidkrad/work/i3kiss/i3setup_extended.sh",
    )  # TODO here pass venv which includes b2luigi

    # Most likely your executable from the submission node is not the same on
    # the worker node, so specify it explicitly
    # b2luigi.set_setting("executable", ["python3"])

    # Where to store the results
    # b2luigi.set_setting("result_dir", "results")
    WORKERS = 10000 if BATCH else 5
    b2luigi.process(Wrapper(), batch=BATCH, workers=WORKERS)
