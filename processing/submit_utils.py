import defaults
from utils import split_list_in_chunks, natural_sort
from pprint import pprint
import os
import glob
import re
import pickle

# input_infos = {
#     "generated": {
#         "directory": "/data/sim/IceCube/2020/generated/CORSIKA-in-ice/{dataset}/{run_range}/generated/",
#         "filename": "prop.corsika.{dataset:06}.{run}.i3.zst",
#         "store_weight_info": True,
#     },
#     "detector": {
#         "directory": "/data/sim/IceCube/2020/generated/CORSIKA-in-ice/{dataset}/{run_range}/detector/",
#         "filename": "IC86.2020_corsika.{dataset:06}.{run}.i3.zst",
#     },
#     "filtered": {
#         "directory": "/data/sim/IceCube/2020/filtered/level2/CORSIKA-in-ice/{dataset}/{run_range}/",
#         "filename": "Level2_IC86.2020_corsika.{dataset:06}.{run}.i3.zst",
#     },
#     "Level3": {
#         "directory": "/data/user/navidkrad/data/sim/IceCube/2020/filtered/level3/CORSIKA-in-ice/{dataset}/{run_range}/",
#         "filename": "Level3_IC86.2020_corsika.{dataset:06}.{run}.i3.zst",
#     },
# }

input_info_dict = {
    23123: {
        "generated": {
            "directory": "/data/sim/IceCube/2023/generated/CORSIKA-in-ice/{dataset}/{run_range}/generated/",
            "filename": "corsika.{dataset:06}.{run}.i3.zst",
            "store_weight_info": True,
        },
        "detector": {
            "directory": "/data/sim/IceCube/2023/generated/CORSIKA-in-ice/{dataset}/{run_range}",
            "filename": "IC86.2023_corsika.{dataset:06}.{run}.i3.zst",
        },
        "filtered": {
            "directory": "/data/sim/IceCube/2023/filtered/level2/CORSIKA-in-ice/{dataset}/{run_range}/",
            "filename": "Level2_IC86.2023_corsika.{dataset:06}.{run}.i3.zst",
        },
    },

    20904: {
        "generated": {
            "directory": "/data/sim/IceCube/2020/generated/CORSIKA-in-ice/{dataset}/{run_range}/generated/",
            "filename": "prop.corsika.{dataset:06}.{run}.i3.zst",
            "store_weight_info": True,
        },
        "detector": {
            "directory": "/data/sim/IceCube/2020/generated/CORSIKA-in-ice/{dataset}/{run_range}/detector",
            "filename": "IC86.2020_corsika.{dataset:06}.{run}.i3.zst",
        },
        "filtered": {
            "directory": "/data/sim/IceCube/2020/filtered/level2/CORSIKA-in-ice/{dataset}/{run_range}/",
            "filename": "Level2_IC86.2020_corsika.{dataset:06}.{run}.i3.zst",
        },
    },
}


dataset = 20904
version_tag = "cascades_v0.2_L3"
output_sub_dir = "/sim/IceCube/2020/CORSIKA-in-ice/{dataset}/{run_range}"
output_dir_base = f"/data/user/navidkrad/hdf/{version_tag}/{output_sub_dir}"

print(output_dir_base)


def time_cache(expire_seconds):
    def decorator(func):
        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key based on function arguments
            key = (args, frozenset(kwargs.items()))
            current_time = time.time()
            if key in cache:
                result, timestamp = cache[key]
                # Check if the cached result is still valid
                if current_time - timestamp < expire_seconds:
                    return result
            # Compute the result and cache it with the current timestamp
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            return result

        return wrapper
    return decorator


# def gather_files_old(input_infos, dataset=20904, n_runs=None, match=True, sim_types=None):
#     print("gather files:")
#     pprint(input_infos)
#     sim_types = list(input_infos) if not sim_types else sim_types
#     input_dataset_generated = input_infos["generated"]["directory"]
#     generated_dirs = natural_sort(
#         glob.glob(input_dataset_generated.format(
#             dataset=dataset, run_range="*"))
#     )
#     run_ranges = [re.search(r"(\d+-\d+)", path).group()
#                   for path in generated_dirs]
#     # print(run_ranges)
#     # assert False

#     if isinstance(n_runs, int):
#         run_ranges = run_ranges[:n_runs]
#     elif isinstance(n_runs, (list, tuple)):
#         assert len(n_runs) == 2
#         run_ranges = run_ranges[n_runs[0]: n_runs[1]]

#     input_files = {}
#     print(run_ranges)
#     for run_range in run_ranges:
#         input_files[run_range] = {}
#         for sim_type in sim_types:
#             input_info = input_infos[sim_type]
#             input_path = os.path.join(
#                 input_info["directory"], input_info["filename"])
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
#         n_droped = 0
#         n_kept = 0
#         for run_range in list(input_files):
#             for run in list(input_files[run_range]):
#                 n_kept += 1
#                 if not len(input_files[run_range][run]) == len(sim_types):
#                     print(
#                         f"For {run_range=}, {run= } found only {list(input_files[run_range][run])}. Will Drop it!"
#                     )
#                     input_files[run_range].pop(run)
#                     n_droped += 1
#         print(
#             f'Dropped {n_droped} files out of {n_kept} files ({round(n_droped/n_kept*100,3)} %)')
#     return input_files

datasets_info_dir = os.path.join(
    defaults.default_config['data_base_dir'], 'datasets_info')


def gather_files(dataset=20904, input_infos=None, n_runs=None, match=True, sim_types=None, store_dir=datasets_info_dir, reload=True):
    # input_infos = input_infos if input_infos
    dataset_info_pkl = os.path.join(
        store_dir, f"{dataset}", "input_infos.pkl")
    print(dataset_info_pkl)
    if os.path.isfile(dataset_info_pkl) and reload:
        print(f"Loading input_infos from {dataset_info_pkl}")
        with open(dataset_info_pkl, "rb") as f:
            input_files = pickle.load(f)
        return input_files

    if not input_infos:
        if dataset not in input_info_dict:
            raise ValueError(
                f"dataset {dataset} not in input_info_dict. Please provide input_infos")
        input_infos = input_info_dict[dataset]

    print("gather files:")
    pprint(input_infos)
    sim_types = list(input_infos) if not sim_types else sim_types
    input_dataset_generated = input_infos["generated"]["directory"]
    generated_dirs = natural_sort(
        glob.glob(input_dataset_generated.format(
            dataset=dataset, run_range="*"))
    )
    run_ranges = [re.search(r"(\d+-\d+)", path).group()
                  for path in generated_dirs]
    # print(run_ranges)
    # assert False

    if isinstance(n_runs, int):
        run_ranges = run_ranges[:n_runs]
    elif isinstance(n_runs, (list, tuple)):
        assert len(n_runs) == 2
        run_ranges = run_ranges[n_runs[0]: n_runs[1]]

    input_files = {}
    print(run_ranges)
    for run_range in run_ranges:
        input_files[run_range] = {}
        for sim_type in sim_types:
            input_info = input_infos[sim_type]
            input_path = os.path.join(
                input_info["directory"], input_info["filename"])
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
        n_droped = 0
        n_kept = 0
        for run_range in list(input_files):
            for run in list(input_files[run_range]):
                n_kept += 1
                if not len(input_files[run_range][run]) == len(sim_types):
                    print(
                        f"For {run_range=}, {run= } found only {list(input_files[run_range][run])}. Will Drop it!"
                    )
                    input_files[run_range].pop(run)
                    n_droped += 1
        print(
            f'Dropped {n_droped} files out of {n_kept} files ({round(n_droped/n_kept*100,3)} %)')
    # Save the input_infos to a pickle file
    os.makedirs(os.path.dirname(dataset_info_pkl), exist_ok=True)
    with open(dataset_info_pkl, "wb") as f:
        pickle.dump(input_files, f)
        print(f"Saved input_files to {dataset_info_pkl}")
    return input_files


def count_files(input_files):
    n_files = sum([len(v) for k,v in input_files.items()])
    return n_files

def max_n_files_per_run(input_files):
    n_files = [len(v) for k,v in input_files.items()]
    max_n_files = max(n_files)
    return max_n_files

def get_input_output_split_by_chunk(
    input_infos=None,
    dataset=20904,
    nruns=None,
    nchunks=None,
    nfiles=20,
    output_dir_base=output_dir_base,
):
    print("getting args")
    input_files_all = gather_files(
        dataset=dataset, input_infos=input_infos, n_runs=nruns, match=True
    )
    args = {}
    for run_range in input_files_all:
        run_chunks = list(
            split_list_in_chunks(list(input_files_all[run_range]), nfiles)
        )
        args[run_range] = {}
        n_chunks = len(run_chunks)
        for ichunk, run_chunk in enumerate(run_chunks):
            input_files_chunk = [input_files_all[run_range][run]
                                 for run in run_chunk]
            input_files_gen = [fs["generated"] for fs in input_files_chunk]
            input_files_det = [fs["detector"] for fs in input_files_chunk]
            input_files_filt = [fs["filtered"] for fs in input_files_chunk]
            input_files_l3 = [fs["Level3"] for fs in input_files_chunk]
            # input_files_cscdprebdt = [fs["CscdPreBDT"] for fs in input_files_chunk]
            input_files_cscdbdt = [fs["CscdBDT"] for fs in input_files_chunk]

            # sub_dir = [x.split("detector")[0] for x in input_files_det]
            sub_dir = [os.path.dirname(x).split("detecotr")[0]
                       for x in input_files_det]
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
                # input_files_cscdprebdt=input_files_cscdprebdt,
                input_files_cscdbdt=input_files_cscdbdt,
                output_pattern=output_pattern,
            )

            if nchunks and ichunk >= nchunks:
                print(f"TEST MODE: skipping chunks >= {nchunks}")
                break
    return args
