
## Installation

### Prerequisites

- Python 3.12 or higher is required.

Check your Python version:

```bash
python --version
```

(Optional) Create a new Python environment with venv:

```bash
export TFXKDIR="/tmp/tfxkit-env"
python -m venv $TFXKDIR
source $TFXKDIR/bin/activate
```

### Instal via PyPI

Install tfxkit from TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple tfxkit
```

---

## Quickstart Example

This example uses a config with the `load_breast_cancer` dataset, builds a basic MLP, trains it, and evaluates its performance.
But first setup and download the example dataset and config file:

```bash
tfxkit-setup-example info.example_dir=./example_project/ 
```


The script will create the following items:
 - train file: `./example_project/train.csv`
 - test file: `./example_project/ test.csv`
 - default config: `./example_project/default_config.yaml`
 - primary config: `./example_project/config.yaml`

The primary config (`config.yaml`) builds on top of the `default_config.yaml` by importing it and overriding specific fields such as model parameters, dataset paths, or any other fields which is specified. Then follow the suggested command for running the exmample config file. It should look something like this:

```bash
tfxkit --config-name=config --config-path=/absolute/path/to/example_project/
```

And this is where the customization fun with the Hydra can start. 
All the parameters in the config file can also be modified from the command line:

```bash
tfxkit --config-name=config --config-path=/absolute/path/to/example_project/ \
       info.model_name=new_name
       model.parameters.layers_list=[10,100,20]
       optimizer.function=adamw
```


# Useful Hydra commands:

As you saw in the example config, you can pass on the `config-name` and `config-path` when running `tfxkit`.
Another useful trick is to print the config file being loaded (including the default values) using `--cfg all` flag:

`tfxkit --config-name=config --config-path=/path/to/dir/ --cfg all`

The nice thing about Hydra is that all config options can be modified via the command line as well for example:

`tfxkit model.parameters.hidden_activation="relu"`

If the config item doesn't already exist, it has to be passed in the command line with `+`, e.g.;

`tfxkit +optimizer.ema_momentum=0.9999`

## To Do's

- Implement the `--multirun` functionality of Hydra:
   Hydra also has options for sweeping over different parameters but this is not fully tested within `tfxkit` yet.
   `tfxkit --multirun optimizer.function=keras.optimizers.AdamW,keras.optimizers.Adam`
- Save the result of training (model weights, etc), hyper tuning, and plots in a more consistent way


---

## Feedback & Contributions

See the [GitHub repo](https://github.com/nrad/tfxkit)

---










## Basic Workflow

```bash
tfxkit --config-name=myconfig --config-path=/path/to/config/file
```

1. Define your config file.
2. Select a model builder.
3. Run training or tuning.

## Contents
