
# Quickstart Guide for TFxKit

This quickstart guide will help you get started with TFxKit quickly.

## Installation

Install TFxKit via pip:

```bash
pip install tfxkit
```

## Setup Example Dataset

Download and prepare the example dataset:

```bash
tfxkit-setup-example info.example_dir=./example_project/ 
```

## Run Training

Run the training process with the default configuration:

```bash
tfxkit
```

## Custom Configuration

To run with a custom configuration file:

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
