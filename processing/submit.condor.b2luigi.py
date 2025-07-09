from time import sleep
import os
import subprocess
import sys
import glob
import re

from pprint import pprint

from functools import lru_cache

# from submit.helpers import split_list_in_chunks, natural_sort
from utils import split_list_in_chunks, natural_sort
import utils

try:
    import b2luigi
# except ImportError:
except ModuleNotFoundError:
    # need to point to the b2luigi location
    print("failed to load b2luigi... will append the sys.path")
    sys.path.append("/home/navidkrad/.local/lib/python3.11/site-packages")
    import b2luigi

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
}

dataset = 20904
output_sub_dir = "/sim/IceCube/2020/CORSIKA-in-ice/{dataset}/{run_range}"


sim_types = list(input_infos)

BATCH = True
TEST = False

# BATCH = False
# TEST = True

# /data/sim/IceCube/2020/filtered/level2/CORSIKA-in-ice/20904/0000000-0000999/Level2_IC86.2020_corsika.020904.000000.i3.zst
# input_dataset_dir = "/data/sim/IceCube/2020/generated/CORSIKA-in-ice/20904/"
# version_tag = "mcprimary_v0.7"
version_tag = "mcprimary_v0.10_mu"
version_tag = "mcprimary_v0.12_mu"
version_tag = "mcprimary_v0.13_genweight"
version_tag = "mcprimary_v0.14_filterinfo"
version_tag = "mcprimary_v0.15_sortedmus"
version_tag = "mcprimary_v0.16_preproc"
version_tag = "mcprimary_v0.17_dsfix"
version_tag = "cascades_v0.1_L3"


NFILES = 50  # process 50 times together
NFILES = 20  # process 50 times together

if TEST:
    NCHUNKS = 10
    NFILES = 10
    NRUNS = 1
    version_tag = version_tag + "_TEST"
else:
    NCHUNKS = None
    NRUNS = 50  # number of run directories
    NRUNS = (0, 50)  # number of run directories
    NRUNS = 20  # number of run directories

# if NRUNS or NCHUNKS:
#    version_tag = version_tag + "_TEST"

output_dir_base = f"/data/user/navidkrad/hdf/{version_tag}/{output_sub_dir}"
print(output_dir_base)


ENVSHEL = "/home/navidkrad/work/i3kiss/run_i3_command.sh"


# @lru_cache()
def gather_files(input_infos, dataset=dataset, n_runs=NRUNS, match=True):
    print("gather files:")
    pprint(input_infos)
    input_dataset_generated = input_infos["generated"]["directory"]
    generated_dirs = natural_sort(
        glob.glob(input_dataset_generated.format(dataset=dataset, run_range="*"))
    )
    run_ranges = [re.search(r"(\d+-\d+)", path).group() for path in generated_dirs]

    if isinstance(n_runs, int):
        run_ranges = run_ranges[:n_runs]
    elif isinstance(n_runs, (list, tuple)):
        assert len(n_runs) == 2
        run_ranges = run_ranges[n_runs[0] : n_runs[1]]

    input_files = {}
    print(run_ranges)
    for run_range in run_ranges:
        input_files[run_range] = {}
        for sim_type in sim_types:
            input_info = input_infos[sim_type]
            input_path = os.path.join(input_info["directory"], input_info["filename"])
            input_path = input_path.format(
                run_range=run_range, dataset=dataset, run="*"
            )

            run_range_files = natural_sort(glob.glob(input_path))
            run_pattern = input_infos[sim_type]["filename"].format(
                dataset=dataset, run=r"(\d+)"
            )

            for filename in run_range_files:
                run = re.search(run_pattern, filename).group(
                    1
                )  # if this fails we have a problem
                if run not in input_files[run_range]:
                    input_files[run_range][run] = {}
                input_files[run_range][run][sim_type] = filename
    if match:
        for run_range in list(input_files):
            for run in list(input_files[run_range]):
                if not len(input_files[run_range][run]) == len(sim_types):
                    print(
                        f"For {run_range=}, {run= } found only {list(input_files[run_range][run])}. Will Drop it!"
                    )
                    input_files[run_range].pop(run)
    return input_files


def get_input_output_split_by_chunk():
    print("getting args")
    input_files_all = gather_files(
        input_infos, dataset=dataset, n_runs=NRUNS, match=True
    )
    args = {}
    for run_range in input_files_all:
        run_chunks = list(
            split_list_in_chunks(list(input_files_all[run_range]), NFILES)
        )
        args[run_range] = {}
        n_chunks = len(run_chunks)
        for ichunk, run_chunk in enumerate(run_chunks):
            input_files_chunk = [input_files_all[run_range][run] for run in run_chunk]
            input_files_gen = [fs["generated"] for fs in input_files_chunk]
            input_files_det = [fs["detector"] for fs in input_files_chunk]
            input_files_filt = [fs["filtered"] for fs in input_files_chunk]
            input_files_l3 = [fs["Level3"] for fs in input_files_chunk]

            sub_dir = [x.split("detector")[0] for x in input_files_det]
            assert len(set(sub_dir)) == 1, (
                "Failed to obtain a unique sub-dir from input_files. Got these sub-dirs: %s"
                % sub_dir
            )
            sub_dir = sub_dir[0].lstrip("/data/")

            output_filename = f"{{sim_type}}/{{sim_type}}_{run_range}_{ichunk+1:02}_of_{n_chunks:02}.hdf5"
            output_pattern = output_dir_base.format(
                dataset=dataset, run_range=run_range
            )
            output_pattern = os.path.join(output_pattern, output_filename)
            args[run_range][ichunk] = dict(
                input_files_gen=input_files_gen,
                input_files_det=input_files_det,
                input_files_filt=input_files_filt,
                input_files_l3=input_files_l3,
                output_pattern=output_pattern,
            )

            if NCHUNKS and ichunk >= NCHUNKS:
                print(f"TEST MODE: skipping chunks >= {NCHUNKS}")
                break
    return args


input_output = get_input_output_split_by_chunk()


class SimHDFWriterWrapper(b2luigi.Task):
    """ """

    input_files = b2luigi.ListParameter(hashed=True)
    output_file = b2luigi.Parameter(hashed=True)
    sim_type = b2luigi.Parameter(default="detector")

    script = "/home/navidkrad/work/i3kiss/processing/SimHDFWriter.py"

    def output(self):
        yield self.add_to_output(self.output_file)

    @b2luigi.on_temporary_files
    def run(self):
        output_file_name = self.get_output_file_name(self.output_file)

        self.command = "python {script} --infiles {infiles} --outfile {outfile} --sim_type {sim_type} {extra_args}"
        # extra_args = '--store_weight_info' if self.sim_type=='detector' else ''
        extra_args = "" if self.sim_type == "generated" else "--store_weight_info"
        extra_args = "--store_weight_info"
        self.command = self.command.format(
            script=self.script,
            infiles=" ".join(self.input_files),
            outfile=output_file_name,
            sim_type=self.sim_type,
            extra_args=extra_args,
        )
        print(ENVSHEL + " " + self.command)
        # self.command = 'python /home/navidkrad/work/i3kiss/SimHDFWriter.py --infiles {infiles} --outfile {outfile} --sim_type {sim_type}'
        # self.command = self.command.format(infiles=' '.join(self.input_files), outfile=self.output_file, sim_type=self.sim_type)
        subprocess.check_call(ENVSHEL + " " + self.command, shell=True)


class HDFCombinerWrapper(b2luigi.Task):
    # batch_system = "local"

    input_files_gen = b2luigi.ListParameter(hashed=True)
    input_files_det = b2luigi.ListParameter(hashed=True)
    input_files_filt = b2luigi.ListParameter(hashed=True)
    input_files_l3 = b2luigi.ListParameter(hashed=True)
    output_pattern = b2luigi.Parameter(hashed=True)
    preproc = b2luigi.BoolParameter(default=False)

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
        # self.output_file_comb = self.output_pattern.format(sim_type="combined")
        self.output_file_comb = self.output_pattern.format(
            sim_type="combined" + ("_preproc" if self.preproc else "")
        )

    def requires(self):
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
        )

    def output(self):
        yield self.add_to_output(self.output_file_comb)

    @b2luigi.on_temporary_files
    def run(self):
        # with open(self.get_output_file_name("test.txt"), "w") as f:
        #    f.write(f"{self.output_det}\n{self.output_gen}")
        # self.command = 'python /home/navidkrad/work/i3kiss/SimHDFWriter.py --infiles {infiles} --outfile {outfile} --sim_type {sim_type}'
        output_file_name = self.get_output_file_name(
            self.output_file_comb
        )  # handles temp file creation
        self.command = f"python {self.script}"
        command_args = [
            f"--input_file_gen  {self.hdf5_file_gen}",
            f"--input_file_det  {self.hdf5_file_det}",
            f"--input_file_filt {self.hdf5_file_filt}",
            f"--input_file_l3   {self.hdf5_file_l3}",
            f"--output_file    {output_file_name}",
            f'--columns {" ".join(self.columns)}',
            f'--index   {" ".join(self.index)}',
            f"--key     {self.key}",
            f"--gen_weights_nfiles {NFILES}",
        ]
        if self.preproc:
            command_args.append("--preproc")
        # self.command = self.command.format(infiles=' '.join(self.input_files), outfile=self.output_file, sim_type=self.sim_type)
        self.command = self.command + " " + " ".join(command_args)
        print(self.command)
        subprocess.check_call(ENVSHEL + " " + self.command, shell=True)

        # with open(self.get_output_file_name(self.output_file_comb), "w") as f:
        #    f.write("combined")


class PreprocBalancerTask(b2luigi.Task):
    sim_type = b2luigi.Parameter(default="combined")
    preselection = b2luigi.Parameter()
    balance_by = b2luigi.Parameter(default="filtered")
    run_range = b2luigi.Parameter()
    do_preproc = b2luigi.BoolParameter()

    selections = {
        "triggered": "triggered==1",
        "hasphotons": "n_photons>0",
        "hasmu": "multiplicity>0",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_tag = f"{self.sim_type}_preproc"
        if self.preselection:
            output_tag = output_tag + f"_presel_{self.preselection}"
            self.presel_query = self.selections[self.preselection]

        if self.balance_by:
            output_tag = output_tag + f"_balanced_by_{self.balance_by}"

        self.output_tag = output_tag

        output_file = os.path.join(
            output_dir_base, "{sim_type}/{sim_type}_{run_range}.hdf5"
        ).format(
            dataset=dataset,
            run_range=self.run_range,
            sim_type=self.output_tag,
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
        import itertools

        input_files = itertools.chain(*list(self.get_input_file_names().values()))
        input_files = list(input_files)
        # input_files = input_output[self.run_range]

        print(f"PreprocBalancerTask: {input_files = }")
        combine_and_preproc(
            input_files,
            output=self.output_file,
            balance_by=self.balance_by,
            preselection=self.presel_query if self.preselection else None,
            do_preproc=self.do_preproc,
        )


class Wrapper(b2luigi.Task):
    batch_system = "local"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def requires(self):
        for run_range in input_output:
            # for ichunk in input_output[run_range]:
            #     yield HDFCombinerWrapper(**input_output[run_range][ichunk])

            for preselection, balance_by, do_preproc in [
                # ("triggered", "filtered"),
                # (None, "filtered", False),
                # (None, "triggered"),
                # (None, None),
                # ("hasmu", "filtered", None),
                # ("hasphotons", "filtered", None),
                # ("hasmu", "Level3", None),
                ("hasphotons", "Level3", None),
                ("hasphotons", "filtered", None),
                # ("hasmu", None, False),
                # ("hasphotons", "filtered", False),
            ]:
                yield PreprocBalancerTask(
                    sim_type="combined",
                    preselection=preselection,
                    balance_by=balance_by,
                    run_range=run_range,
                    do_preproc=do_preproc,
                )

    # def output(self):
    #     input_output = self.input_output
    #     sim_type = "combined_preproc_balanced"

    # @b2luigi.on_temporary_files
    # def run(self):
    #     for run_range in self.input_output:
    #         inputs = self.input_output[run_range]
    #     self.command = self.command + " " + " ".join(command_args)
    #     print(self.command)
    #     subprocess.check_call(ENVSHEL + " " + self.command, shell=True)

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
            "Request_Memory": "8GB",
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

    b2luigi.process(Wrapper(), batch=BATCH, workers=1000 if BATCH else 5)
