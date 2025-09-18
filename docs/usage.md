
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

### Installation

Install tfxkit from TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple tfxkit
```

---

## 🚀 Quickstart Example

This example uses a config with the `load_breast_cancer` dataset, builds a basic MLP, trains it, and evaluates its performance.
But first setup and download the example dataset and config file:

```bash
tfxkit-setup-example
```


The script will create the following items:
 - train file: `$TFXKDIR/examples/train.csv`
 - test file: `$TFXKDIR/examples/test.csv`
 - default config: `$TFXKDIR/examples/default_config.yaml`
 - primary config: `$TFXKDIR/examples/example.yaml`

The primary config (example.yaml) builds on top of the default_config.yaml by importing it and overriding specific fields such as model parameters, dataset paths, or any other fields which is specified. 

Then follow the suggested command for running the exmample config file. It should look something like this:

```bash
tfxkit --config-name=example --config-path=$TFXKDIR/examples
```



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
