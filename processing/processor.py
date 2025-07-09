import submit_utils
from copy import deepcopy
import os
import tempfile
import subprocess


class FormatListTemplate(str):
    def format(self, **kwargs):
        list_keys = [k for k, v in kwargs.items() if isinstance(v, list)]
        if not list_keys:
            return str.format(self, **kwargs)  # Direct call to str.format
        key = list_keys[0]
        values = kwargs[key]
        return [str.format(self, **{**kwargs, key: val}) for val in values]


steering_input_2023 = {
    "generated_output": "corsika.{dataset:06g}.{run}.i3.zst",
    "triggered_output": "IC86.2023_corsika.{dataset:06g}.{run}.i3.zst",
    "filtered_output": "Level2_IC86.2023_corsika.{dataset:06g}.{run}.i3.zst",
    "level3_output": "Level3_IC86.2023_corsika.{dataset:06g}.{run_combined}.i3.zst",
    "cscdPreBDT_output": "CscdSBU_PreBDT_IC86.2023_corsika.{dataset:06g}.{run_combined}.i3.zst",
    "cscdBDT_output": "CscdSBU_BDT_IC86.2023_corsika.{dataset:06g}.{run_combined}.i3.zst",
    "filtered_output_dir": "/data/sim/IceCube/2023/filtered/level2/CORSIKA-in-ice/{dataset}/{run_range}/",
    "triggered_output_dir": "/data/sim/IceCube/2023/generated/CORSIKA-in-ice/{dataset}/{run_range}/",
    "generated_output_dir": "/data/sim/IceCube/2023/generated/CORSIKA-in-ice/{dataset}/{run_range}/generated/",
}

steering_input_2020 = {
    "generated_output": "prop.corsika.{dataset:06g}.{run}.i3.zst",
    "triggered_output": "IC86.2020_corsika.{dataset:06g}.{run}.i3.zst",
    "filtered_output": "Level2_IC86.2020_corsika.{dataset:06g}.{run}.i3.zst",
    "level3_output": "Level3_IC86.2020_corsika.{dataset:06g}.{run_combined}.i3.zst",
    "cscdPreBDT_output": "CscdSBU_PreBDT_IC86.2020_corsika.{dataset:06g}.{run_combined}.i3.zst",
    "cscdBDT_output": "CscdSBU_BDT_IC86.2020_corsika.{dataset:06g}.{run_combined}.i3.zst",
    "filtered_output_dir": "/data/sim/IceCube/2020/filtered/level2/CORSIKA-in-ice/{dataset}/{run_range}/",
    "triggered_output_dir": "/data/sim/IceCube/2020/generated/CORSIKA-in-ice/{dataset}/{run_range}/detector/",
    "generated_output_dir": "/data/sim/IceCube/2020/generated/CORSIKA-in-ice/{dataset}/{run_range}/generated/",
}


steering_common = {
    "generated_hdf_output": "generated_{run_combined}.hdf5",
    "triggered_hdf_output": "triggered_{run_combined}.hdf5",
    "filtered_hdf_output": "filtered_{run_combined}.hdf5",
    "level3_hdf_output": "Level3_{run_combined}.hdf5",
    "cscdbdt_hdf_output": "CscdBDT_{run_combined}.hdf5",
    "combiner_hdf_output": "combined_{run_combined}.hdf5",
    # 'acceptance_probability': '0.1',
    "acceptance_probability": "0.001",
    # 'n_files': '9739',  # number of files in the generated dataset
    "temp_output_directory": tempfile.gettempdir(),
    "output_directory": "/path/to/final/output/directory/",
    # 'icetray-envshell': '/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_7_x86_64/metaprojects/icetray/v1.12.0/env-shell.sh',
    # including utils
    "icetray-envshell": "/home/navidkrad/scripts/job_runners/py3-v4.3.0_tf2p17.sh",
    "sbu-env": "/cvmfs/icecube.opensciencegrid.org/users/zzhang1/combo_stable_addatmo/env-shell.sh",
}

steering_datasets = {
    20904: dict(
        **steering_input_2020,
        n_files="99874",
    ),
    23123: dict(
        **steering_input_2023,
        n_files="11931",
    ),
}


def get_config(steering):
    config = [
        {
            "name": "CascadeFilter",
            #  'input_dir': '/data/sim/IceCube/2023/filtered/level2/CORSIKA-in-ice/{dataset}/',
            "output": os.path.join(
                steering["temp_output_directory"], steering["level3_output"]
            ),
            "input": os.path.join(
                steering["filtered_output_dir"], steering["filtered_output"]
            ),
            #  'command': 'python {I3_BUILD}/lib/icecube/level3_filter_cascade/level3_Master.py --input {input_files} --gcd {GCDFILE} -o {output_file} --MC',
            "seperator": ",",
            "command": [
                steering["icetray-envshell"],
                "python",
                #  '/home/navidkrad/work/i3kiss/processing/level3_Master.py',
                "{I3_BUILD}/lib/icecube/level3_filter_cascade/level3_Master.py",
                "--input",
                "{input_files}",
                "--gcd",
                "{GCDFILE}",
                "-o",
                "{output_file}",
                "--MC",
                # '&',
            ],
        },
        {
            "name": "SBU_CscdPreBDT",
            "input": os.path.join(
                steering["temp_output_directory"], steering["level3_output"]
            ),
            "output": os.path.join(
                steering["temp_output_directory"], steering["cscdPreBDT_output"]
            ),
            "command": [
                steering["sbu-env"],
                "python",
                "/home/navidkrad/work/wg-diffuse/cascade-final-filter/cscdSBU_master.py",
                "-i",
                "{input_files}",
                "-o",
                "{output_file}",
                "--g",
                "{GCDFILE}",
                "-d",
                "cor",
                "--year",
                "2022",
            ],
            "requires": "CascadeFilter",
        },
        {
            "name": "SBU_CscdBDT",
            "input": os.path.join(
                steering["temp_output_directory"], steering["cscdPreBDT_output"]
            ),
            "output": os.path.join(
                steering["temp_output_directory"], steering["cscdBDT_output"]
            ),
            "command": [
                "/home/navidkrad/work/wg-diffuse/cascade-final-filter/selection/start_nonecut_local.sh",
                "{GCDFILE}",
                "{input_files}",
                "{output_file}",
                "cors",
                "12345",
                "0",
            ],
            "requires": "SBU_CscdPreBDT",
        },
        {
            "name": "SBU_CscdBDTHDF",
            "input": os.path.join(
                steering["temp_output_directory"], steering["cscdBDT_output"]
            ),
            "output": os.path.join(
                steering["temp_output_directory"], steering["cscdbdt_hdf_output"]
            ),
            "command": [
                steering["icetray-envshell"],
                "python",
                "/home/navidkrad/work/i3kiss/processing/SimHDFWriter.py",
                "--infiles",
                "{input_files}",
                "--outfile",
                "{output_file}",
                "--sim_type",
                "CscdBDT",
            ],
            "requires": "SBU_CscdBDT",
        },
        {
            "name": "GeneratedHDF",
            #  'input_dir': '/data/sim/IceCube/2023/generated/CORSIKA-in-ice/{dataset}/generated/',
            "input": os.path.join(
                steering["generated_output_dir"], steering["generated_output"]
            ),
            "output": os.path.join(
                steering["temp_output_directory"], steering["generated_hdf_output"]
            ),
            "command": [
                steering["icetray-envshell"],
                "python",
                "/home/navidkrad/work/i3kiss/processing/SimHDFWriter.py",
                "--infiles",
                "{input_files}",
                "--outfile",
                "{output_file}",
                "--store_weight_info",
                "--sim_type",
                "generated",
                "--count_photons",
                # '--split_muons',
                "--add_shower_muons",
                "--acceptance_probability",
                steering["acceptance_probability"],
                "--target_primary_files",
                os.path.join(
                    steering["filtered_output_dir"], steering["filtered_output"]
                ),
            ],
            "requires": "CascadeFilter",
        },
        {
            "name": "TriggeredHDF",
            #  'input_dir': '/data/sim/IceCube/2023/generated/CORSIKA-in-ice/{dataset}/',
            "input": os.path.join(
                steering["triggered_output_dir"], steering["triggered_output"]
            ),
            "output": os.path.join(
                steering["temp_output_directory"], steering["triggered_hdf_output"]
            ),
            "command": [
                steering["icetray-envshell"],
                "python",
                "/home/navidkrad/work/i3kiss/processing/SimHDFWriter.py",
                "--infiles",
                "{input_files}",
                "--outfile",
                "{output_file}",
                "--sim_type",
                "detector",
            ],
        },
        {
            "name": "FilteredHDF",
            #  'input_dir': '/data/sim/IceCube/2023/filtered/level2/CORSIKA-in-ice/{dataset}/',
            "input": os.path.join(
                steering["filtered_output_dir"], steering["filtered_output"]
            ),
            "output": os.path.join(
                steering["temp_output_directory"], steering["filtered_hdf_output"]
            ),
            "command": [
                steering["icetray-envshell"],
                "python",
                "/home/navidkrad/work/i3kiss/processing/SimHDFWriter.py",
                "--infiles",
                "{input_files}",
                "--outfile",
                "{output_file}",
                "--sim_type",
                "filtered",
            ],
        },
        {
            "name": "Level3HDF",
            # 'input': os.path.join(steering['level3_output_dir'], steering['level3_output']),
            "input": os.path.join(
                steering["temp_output_directory"], steering["level3_output"]
            ),
            "output": os.path.join(
                steering["temp_output_directory"], steering["level3_hdf_output"]
            ),
            "command": [
                steering["icetray-envshell"],
                "python",
                "/home/navidkrad/work/i3kiss/processing/SimHDFWriter.py",
                "--infiles",
                "{input_files}",
                "--outfile",
                "{output_file}",
                "--sim_type",
                "Level3",
            ],
            "requires": "CascadeFilter",
        },
        {
            "name": "CombinerHDF",
            "input_key": "",
            "output": os.path.join(
                steering["temp_output_directory"], steering["combiner_hdf_output"]
            ),
            "command": [
                steering["icetray-envshell"],
                "python",
                "/home/navidkrad/work/i3kiss/processing/CombineGenDet.py",
                "--input_file_gen",
                # '{input_files_gen}',
                os.path.join(
                    steering["temp_output_directory"], steering["generated_hdf_output"]
                ),
                "--input_file_det",
                # '{input_files_det}',
                os.path.join(
                    steering["temp_output_directory"], steering["triggered_hdf_output"]
                ),
                "--input_file_filt",
                # '{input_files_filt}',
                os.path.join(
                    steering["temp_output_directory"], steering["filtered_hdf_output"]
                ),
                "--input_file_l3",
                os.path.join(
                    steering["temp_output_directory"], steering["level3_hdf_output"]
                ),
                # '{input_files_l3}',
                "--input_file_cscdbdt",
                os.path.join(
                    steering["temp_output_directory"], steering["cscdbdt_hdf_output"]
                ),
                # '{input_files_cscdbdt}',
                "--output_file",
                os.path.join(
                    steering["temp_output_directory"], steering["combiner_hdf_output"]
                ),
                "--add_x_mins",
                "--columns",
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
                "--index",
                "energy",
                "minorID",
                "z",
                "--key",
                "PolyplopiaPrimary",
                "--gen_weights_nfiles",
                steering["n_files"],
                "--keep_filters",
                "--selection_fractions",
                "generated",
                "0.01",
                "triggered",
                "0.01",
                "filtered",
                "1.0",
                "Level3",
                "1.0",
                "CscdBDT",
                "1.0",
                "--selection_weight_key",
                "selection_weights_partial",
                "--preproc",
            ],
        },
        {
            "name": "Mkdir",
            "output": steering["output_directory"],
            "command": ["mkdir", "-p", "{output_file}"],
        },
        {
            "name": "OutputTransfer",
            "input": os.path.join(
                steering["temp_output_directory"], steering["combiner_hdf_output"]
            ),
            "output": os.path.join(
                steering["output_directory"], steering["combiner_hdf_output"]
            ),
            "command": ["mv", "{input_files}", "{output_file}"],
        },
    ]
    return config


common_dict = {
    "I3_BUILD": "/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_7_x86_64/metaprojects/icetray/v1.8.2",
    "GCDFILE": "/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz",
}


def _check_unformatted(string):
    if "{" in string or "}" in string:
        raise ValueError(f"String not formatted correctly: {string}")
    return True


def get_input_file_list(
    template, dataset, run_range, runs, return_list=False, seperator=" "
):
    runs = runs if isinstance(runs, (tuple, list)) else [runs]
    input_files = []
    for run in runs:
        input_file = template.format(
            dataset=dataset, run_range=run_range, run=f"{run:06g}"
        )
        # if not os.path.exists(input_file):
        #     raise ValueError(f'Input file not found: {input_file}')
        input_files.append(input_file)
    if return_list:
        return input_files
    else:
        return seperator.join(input_files)


def get_output_file(template, dataset, run_range, runs):
    runs = runs if isinstance(runs, (tuple, list)) else [runs]
    run_tag = f"{runs[0]:06g}" if len(runs) == 1 else f"{runs[0]:06g}_{runs[-1]:06g}"
    # print(f'{dataset=}, {run_range=}, {runs=} {run_tag=}')
    # print(template)
    output_file = template.format(dataset=dataset, run_range=run_range, run=run_tag)
    return output_file


def format_filename(template, dataset, run_range, runs, verbose=False, seperator=" "):
    if verbose:
        print("================")
        print(f"{template=}")
        print(f"{dataset=}, {run_range=}, {runs=}")
    if r"{run_combined}" in template:
        # print('getting output')
        out = get_output_file(
            template.replace("{run_combined}", "{run}"), dataset, run_range, runs
        )
    elif r"{run}" in template:
        out = get_input_file_list(
            template, dataset, run_range, runs, seperator=seperator
        )
    else:
        out = template.format(dataset=dataset, run_range=run_range)
    if verbose:
        print(f"{out = }")
        print("================")
    return out


def create_commands(dataset, run_range, runs, config):

    data_run_info = dict(dataset=dataset, run_range=run_range, runs=runs)

    commands = [
        "#!/bin/bash",
        "set -e",
    ]

    command_dicts = []

    for step_config in config:
        # print('------------', '\n'+step_config['name'])
        step_name = step_config["name"]
        # print(step_config)

        command_dict = {}
        # command_dict = deepcopy(step_config.get('command_dict', {}))
        # for k, v in command_dict.items():
        #     if isinstance(v, str):
        #         command_dict[k] = v.format(
        #             dataset=dataset, run_range=run_range, runs=runs)
        # print('command_dict', command_dict)

        # output_file = step_config['output']
        # output_file = output_file.format(
        #     dataset=dataset, run_range=run_range, run=run)
        # output_file
        output_file = format_filename(step_config["output"], dataset, run_range, runs)
        command_dict["output_file"] = output_file

        # print(f'{output_file=}')
        if "{" in output_file or "}" in output_file:
            raise ValueError(f"output_file not formatted correctly: {output_file}")

        # print('output_file', output_file)

        if "input" in step_config:
            input_file = step_config["input"]
            seperator = step_config.get("seperator", " ")
            # input_file = input_file.format(
            #     dataset=dataset, run_range=run_range, run=run)
            command_dict["input_files"] = format_filename(
                input_file, dataset, run_range, runs, seperator=seperator
            )

        command_dict.update(data_run_info)

        for k, v in common_dict.items():
            command_dict.setdefault(k, v)

        # print('command_dict', command_dict)
        command = step_config["command"]

        fixed_command = []
        if isinstance(command, (list, tuple)):
            for command_part in command:
                # print(command_part)
                if "{run}" in command_part or "{run_combined}" in command_part:
                    # print(f'HAS RUN: {command_part}')
                    fnames = format_filename(command_part, dataset, run_range, runs)
                    # print('command_part', command_part)
                    # fixed_command.append(''.join(fnames))
                    fixed_command.append(fnames)
                else:
                    fixed_command.append(command_part)
            command = fixed_command
            # print(f'fixed! {fixed_command=}')

        command = command if isinstance(command, str) else " ".join(command)
        # print(command_dict)
        print("command", command)
        command = command.format(**command_dict)
        _check_unformatted(command)

        commands.append("echo ")
        commands.append("echo -----------------------------")
        commands.append("echo ")
        commands.append(f"# {step_config['name']}")
        commands.append(f'echo {step_config["name"]}')
        commands.append(f'echo "{command}"')
        commands.append("time " + command)
        commands.append("echo -----------------------------")
        # print('command:\n', command)
    return commands


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int)
    parser.add_argument("--run_range", type=str)
    parser.add_argument("--runs", nargs="+", type=int)
    parser.add_argument("--processing_tag", type=str, default="TEST")
    parser.add_argument(
        "--output_directory",
        type=str,
        default="/data/sim/IceCube/2023/filtered/level3/CORSIKA-in-ice/",
    )
    parser.add_argument("--execute", action="store_true")
    # parser.add_argument('--mode', type=str, choices=['execute'] )
    parser.add_argument("--make_commands", action="store_true")
    parser.add_argument("--n_chunks", type=int, default=1)
    args = parser.parse_args()

    dataset = args.dataset
    run_range = args.run_range
    runs = args.runs

    processing_tag = args.processing_tag

    steering = steering_datasets[dataset]
    steering.update(steering_common)

    output_base_directory = f"/data/user/navidkrad/hdf/{processing_tag}/sim/"
    output_directory = os.path.join(
        f"{output_base_directory}", "{dataset}/{run_range}/combined_preproc_presel/"
    )
    steering["output_directory"] = output_directory

    config = get_config(steering)

    if args.execute and args.make_commands:
        raise ValueError("Cannot use --execute and --make_commands at the same time")

    if args.make_commands:
        from utils import split_list_in_chunks

        input_files = submit_utils.gather_files(dataset=dataset)
        # n_files = submit_utils.gather_files(input_files)
        # steering['n_files'] = n_files
        max_n_files = submit_utils.max_n_files_per_run(input_files)

        final_output = config[-1]["output"]
        commands = []

        n_done = 0
        n_todo = 0

        for run_range in input_files:
            run_list = list(input_files[run_range])
            run_i, run_f = run_range.split("-")
            chunk_list = split_list_in_chunks(
                ["{:06g}".format(k) for k in range(int(run_i), int(run_f) + 1)],
                args.n_chunks,
            )
            chunk_list = list(chunk_list)
            # print(chunk_list)
            # print(chunk_list)
            # chunk_list = [k for k in chunk_list if k in run_list]
            # chunk_list = split_list_in_chunks(run_list, args.n_chunks)
            # print(chunk_list)
            # print(run_list)
            # print(run_i, run_f, run_range, run_list)
            for chunk in chunk_list:
                chunk_expected = chunk[:]
                # print('before', len(chunk),)
                # chunk_2 = [r for r in chunk if r in run_list]
                # if not len(chunk_2):
                #     print(chunk)
                #     print(run_list)
                chunk = [r for r in chunk if r in run_list]
                # print('after', len(chunk),)
                # print('--------------')
                if not chunk:
                    print(f"WARNING: No files for runs: {chunk_expected}")
                    continue
                runs = " ".join([str(run) for run in chunk])
                command = f"/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_7_x86_64/bin/python /home/navidkrad/work/i3kiss/processing/processor.py --dataset {dataset} --run_range {run_range} --runs {runs} --processing_tag {processing_tag} --execute"
                output_file = format_filename(
                    final_output, dataset, run_range, [int(k) for k in chunk]
                )

                if not os.path.isfile(output_file):
                    commands.append(command)
                    n_todo += 1
                    print(f"final output to do: {n_todo}: {output_file=}")
                else:
                    n_done += 1
                    # print(f"Already done: {output_file}")

        fname_tag = f"{processing_tag}_{dataset}_{n_todo}_of_{n_done+n_todo}"
        commands_output_file = f"jobs_processor_{fname_tag}.sh"
        print(f"Already done: {n_done} out of {n_done+n_todo}")
        print(len(commands), "jobs to do")
        print("\n".join(commands), file=open(commands_output_file, "w"))
        print(f"Commands written to:\n{commands_output_file}")
        exit()
    else:
        commands = create_commands(dataset, run_range, runs, config)
        run_tag = (
            f"{runs[0]:06g}" if len(runs) == 1 else f'{runs[0]:06g}_{runs[-1]:06g}'
        command_file = os.path.join(tempfile.gettempdir(
        ), f'commands_{processing_tag}_{dataset}_{run_range}_{run_tag}.sh')

    if args.execute:
        print("\n".join(commands), file=open(command_file, 'w'))
        print('running commands: \n', command_file)
        print('-------------------------')
        subprocess.run(['bash', command_file])
    else:
        print("\n".join(commands))
