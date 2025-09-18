# Usage Guide

This guide explains how to use `tfxkit` for training and tuning.

## Prerequisites

- Python 3.12 or higher is required.

Check your Python version:

```bash
python --version
```

(Optional) Create a new Python environment with venv:

```bash
python -m venv tfxkit-env
source tfxkit-env/bin/activate
```

## Installation

Install tfxkit from TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple tfxkit
```

Install from source:

```bash
git clone git@github.com:nrad/tfxkit.git
cd tfxkit
pip install -e .
```

## Run the Example

First setup and download the example dataset and config file:

```bash
tfxkit-setup-example
```

Then you can run training with the default configuration:

```bash
tfxkit
```

## Basic Workflow

```bash
tfxkit --config-name=myconfig --config-path=/path/to/config/file
```

1. Define your config file.
2. Select a model builder.
3. Run training or tuning.

## Contents

- [Quickstart](quickstart.md)