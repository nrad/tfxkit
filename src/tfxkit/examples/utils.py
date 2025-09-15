from importlib.resources import files
import shutil


def download_example_data(dir=None):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer(as_frame=True)
    df_train, df_test = train_test_split(data.frame, test_size=0.3, random_state=42)
    for df, name in [(df_test, "test"), (df_train, "train")]:
        df.reset_index(drop=True, inplace=True)
        df = df.sample(frac=1)
        file_path = (
            files("tfxkit.examples").joinpath(f"{name}.csv")
            if dir is None
            else f"{dir}/{name}.csv"
        )
        df.to_csv(file_path, index=False)


def create_example_config():
    template = """
docs: |
  Quickstart configuration for TFXKit experiments.
  This is example configuation based on the breast cancer dataset of sklearn.
  It includes a simple model architecture, basic data settings, and a short training duration.
  You can modify this configuration to suit your specific needs.

defaults:
  - config

data:
  file_reader: tfxkit.common.base_utils.read_csv
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

tuner:
  functions:
    generic_tuning:
      parameters:
        model.parameters.layers_list: [ "[64,32]", "[32, 64]", "[16,32]", "64,128,64,32" ]
        model.parameters.hidden_activation: ["relu", "tanh"]
        optimizer.function: ["keras.optimizers.Adam", "keras.optimizers.AdamW"]
        optimizer.parameters.learning_rate: [0.001, 0.0001]
        #kernel_initializer:
        optimizer.parameters.weight_decay: [0.01, 0.001]
        model.parameters.dropout: [0.2, 0.5, 0.8]


"""
    example_dir = files("tfxkit.examples")
    # config_path = os.path.join(example_dir, "example_config.yaml")
    config_path = example_dir / "example.yaml"
    with open(config_path, "w") as f:
        f.write(
            template.format(
                test_file=example_dir.joinpath("test.csv"),
                train_file=example_dir.joinpath("train.csv"),
            )
        )

    print(f"Example config file created at: {config_path}")


def setup_example(example_dir=None):
    download_example_data(dir=example_dir)
    create_example_config()
