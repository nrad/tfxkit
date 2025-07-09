import os
import glob
from utils import natural_sort
import argparse
# Define base directories

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
args = parser.parse_args()
dataset = args.dataset


input_dir_base = "/data/sim/IceCube/2023/filtered/level2/CORSIKA-in-ice/{dataset}/".format(dataset=dataset)

N_RUNS = 101

input_dirs = natural_sort(glob.glob(os.path.join(input_dir_base, "*")))
input_dirs = input_dirs[:N_RUNS]


output_dir_base = "/data/user/navidkrad/"

# Define the GCD file path and I3_BUILD variable (set these according to your environment)
GCDFILE = os.getenv("GCDFILE", "GCDFILEPATH")
I3_BUILD = os.getenv("I3_BUILD", None)
if not I3_BUILD:
    raise ValueError("I3_BUILD environment variable not set... set I3 environment first")


mkdir_commands = []
done = []
commands = []

incomp = []
for input_dir in input_dirs:
    # Modify output directory based on input directory, replacing level2 with level3
    output_dir = os.path.join(
        output_dir_base, input_dir.lstrip("/").replace("level2", "level3")
    )

    # Uncomment the line below to actually create the output directory
    # os.makedirs(output_dir, exist_ok=True)
    if not os.path.isdir(output_dir):
        mkdir_commands.append(f"mkdir {output_dir}")

    # Loop over .i3.zst files in the input directory
    for input_file in glob.glob(os.path.join(input_dir, "*.i3.zst")):
        out_filename = os.path.basename(input_file).replace("Level2", "Level3")
        out_file = os.path.join(output_dir, out_filename)

        # Check if the output file already exists and is not empty
        command = f"python {I3_BUILD}/lib/icecube/level3_filter_cascade/level3_Master.py --input {input_file} --gcd {GCDFILE} -o {out_file} --MC"
        if os.path.exists(out_file):
            if os.path.getsize(out_file) > 0:
                done.append(command)
                continue
            else:
                incomp.append(out_file)

        commands.append(command)

print("\n".join(commands))
print("#mkdir commands")
print("\n".join(mkdir_commands))
print(f"# already done {len(done)}/{len(commands)}")
