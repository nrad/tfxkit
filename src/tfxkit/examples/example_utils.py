import logging
from importlib.resources import files
import os
import shutil
from omegaconf import OmegaConf, open_dict
import yaml

logger = logging.getLogger(__name__)


def download_example_data(output_dir=None):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer(as_frame=True)
    df_train, df_test = train_test_split(data.frame, test_size=0.3, random_state=42)
    for df, name in [(df_test, "test"), (df_train, "train")]:
        df.reset_index(drop=True, inplace=True)
        df = df.sample(frac=1)
        file_path = (
            files("tfxkit.examples").joinpath(f"{name}.csv")
            if output_dir is None
            else f"{output_dir}/{name}.csv"
        )
        logging.info(f"\n{name} file saved in:\n{file_path}\n")
        df.to_csv(file_path, index=False)


def create_example_config(default_cfg=None, example_dir=None):
    template = """
docs: |
  Quickstart configuration for TFXKit experiments.
  This is example configuation based on the breast cancer dataset of sklearn.
  It includes a simple model architecture, basic data settings, and a short training duration.
  You can modify this configuration to suit your specific needs.
  The default values are taken from the default_config.yaml file, and the values here are overrides.

defaults:
  - default_config
  - _self_

info:
  model_name: example_model
  save_dir: {example_dir}

data:
  file_reader: tfxkit.common.base_utils.default_reader
  features:
    - mean radius
    - mean texture
    - mean perimeter
    - mean area
    - mean smoothness
  labels: ["target"]

  train_files: {train_file}
  test_files: {test_file}
  train_weights_column: null

model:
  parameters:
    hidden_activation: relu
    final_activation: sigmoid
    layers_list: [64, 32, 16]
    batch_norm_features: true
    batch_norm_hidden: true
    dropout: 0.8

training:
  epochs: 10
  batch_size: 32
  validation_split: 0.2

plotter:
  plots_path: ${{info.save_dir}}/${{info.model_name}}/plots # will be set to info.save_dir/info.model_name/plots

tuner:
  functions:
    generic_tuning:
      parameters:
        model.parameters.layers_list: [ "[64,32]", "[32, 64]", "[16,32]", "64,128,64,32" ]
        model.parameters.hidden_activation: ["relu", "tanh"]
        optimizer.function: ["keras.optimizers.Adam", "keras.optimizers.AdamW"]
        optimizer.parameters.learning_rate: [0.001, 0.0001]
        optimizer.parameters.weight_decay: [0.01, 0.001]
        model.parameters.dropout: [0.2, 0.5, 0.8]

  tuner:
    function: keras_tuner.BayesianOptimization
    parameters:
        alpha: 0.0001
        beta: 2.6
        directory: tfxkit_results/HPTunning/


"""
    # Use provided example_dir or default to package examples directory
    if example_dir is None:
        package_dir = files("tfxkit")
        example_dir = package_dir.joinpath("examples")
    else:
        # Convert string path to Path object if needed
        from pathlib import Path

        example_dir = Path(example_dir)
        # Create directory if it doesn't exist
        example_dir.mkdir(parents=True, exist_ok=True)

    config_filename = "config.yaml"
    config_path = example_dir / config_filename
    with open(config_path, "w") as f:
        f.write(
            template.format(
                test_file=example_dir.joinpath("test.csv"),
                train_file=example_dir.joinpath("train.csv"),
                example_dir=os.path.abspath(example_dir),
            )
        )
    if default_cfg:
        from omegaconf import OmegaConf, open_dict
        import yaml

        with open(example_dir / "default_config.yaml", "w") as f:
            # with open_dict(default_cfg):
            #     for key in ["features", "test_files", "train_files"]:
            #         default_cfg["data"].pop(key, None)
            #     default_cfg.pop("docs")
            #     default_cfg.pop("doc")

            yaml.dump(OmegaConf.to_container(default_cfg), f, width=120, indent=4)

    logging.info(f"Example config file created at: {config_path}")
    config_name = os.path.splitext(config_filename)[0]
    example_dir_absolute_path = os.path.abspath(example_dir)
    command = f"tfxkit --config-name={config_name} --config-path={example_dir_absolute_path}"
    logging.info(
        "\n"
        + "--" * 20
        + "\n"
        + f"Run the example using:\n{command}"
        + "\n"
        + "--" * 20
        + "\n"
    )



def setup_example(default_cfg=None):
    # default_cfg = OmegaConf.load(default_cfg)
    example_dir = None
    if default_cfg:
        with open_dict(default_cfg):
            for key in ["features", "test_files", "train_files"]:
                default_cfg["data"].pop(key, None)
            example_dir = default_cfg["info"].pop("example_dir")
            default_cfg.pop("docs", None)
            default_cfg.pop("doc", None)
    create_example_config(default_cfg=default_cfg, example_dir=example_dir)
    download_example_data(output_dir=example_dir)
