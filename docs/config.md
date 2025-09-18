# TFxKit Configuration Guide

This guide outlines the structure and purpose of each section in a typical TFxKit configuration file. TFxKit uses [Hydra](https://hydra.cc) and [OmegaConf](https://omegaconf.readthedocs.io) for hierarchical configuration management.

## Overview

Each TFxKit config YAML file is composed of several parts:

- `data`: Paths and options related to input datasets, feature and label names etc.
- `model`: Architecture-related choices.
- `optimizer`: Optimizer function and learning rate.
- `training`: Training hyperparameters like batch size and epochs.
- `logging`: Log level settings.
- `tuning` *(optional)*: Setup for hyperparameter search.
- `defaults`: Optional includes or overrides for composing configs.

## Example Config File

### data

Defines where to find the input data and which fields to use.

- `train_file`, `test_file`: Path to CSV or HDF5 files.
- `label`: Name of the target column.
- `features`: Optionally list input feature names.
- `sample_weight_column`: Optional column to use as weights during training.
- `file_reader` this should be the path to a function that takes a list of file paths. It will be called seperately for the test and train files. 
- `xy_maker` this is a function that can perform any additional preprocessing on the test and train files. It should return the `X`, `y` and the optionally `sample_weights`. It will be called seperately for the test and the train datasets.

```yaml
data:
  train_file: examples/train.csv
  test_file: examples/test.csv
  xy_maker: tfxkit.common.tf_utils.xy_maker 
  file_reader: tfxkit.common.base_utils.read_hdfs
  labels: 
    - target
  features: 
    - feature1
    - feature2
    - feature3
  sample_weight_column: weight_column
```

### model

Controls model construction via a builder function.

- `function`: Import path to a model builder function.
- `parameters`: All the key word arguments that can be passed to the above function

For example in the following example the model will be defined as:
`tfxkit.common.tf_utils.define_mlp(**parameters)`

User can of course provide custom function, in which case the parameters must be modified accordingly. 


```yaml
model:
  function: "tfxkit.common.tf_utils.define_mlp"
  parameters:
    hidden_activation: relu
    layers_list: [64, 32, 16]
    final_activation: sigmoid
    batch_norm_features: true
    batch_norm_hidden: true
```

### training

Training hyperparameters.

- `batch_size`
- `epochs`
- `validation_split`: e.g., 0.2 for 80/20 split


```yaml
training:
  batch_size: 81920
  epochs: 10
  validation_split: 0.2
  class_weight: null
  sample_weight: null
```


### optimizer

Specifies the optimizer used in training.

- `function`: e.g., `'adam'`, `'sgd'`
- `learning_rate`: float
```yaml
optimizer:
  function: keras.optimizers.AdamW
  loss: binary_crossentropy
  metrics:
    - accuracy
  parameters:
    learning_rate: 0.001
    weight_decay: 0.0005
    clipnorm: 1.0
    ema_momentum: 0.999
    ema_overwrite_frequency: true
```


### tuning (optional)

This is where things get interesting! 
The first step is to have a `hypermodel` that works with kears_tuner
for more info see for example https://keras.io/keras_tuner/api/tuners/bayesian/.

The different parts of the `model`, `optimizer` and `training` can be hypertuned
using the already defined `generic_tuning` function. For example, by giving a list
to `model.parameters.hidden_activation: ['relu', 'tanh']` these options will be included
as choices during the hypertuning. Note that this is slightly more complicated for the
model architecture. But the different options for `layers_list` can still be provided but
as `str`, just because `keras_tuner.hyperparameter.Choice` does not accept lists. 


```yaml
tuner:
  functions:
    generic_tuning:
      function: "tfxkit.core.tuner.generic_tuner"
      parameters:
        model.parameters.layers_list: [ "[100,200,500]", "[100,200]","[100]"  ]
        model.parameters.hidden_activation: ["relu", "tanh"]
        model.parameters.final_activation: ["sigmoid", "softmax"]
        optimizer.function: ["keras.optimizers.Adam", "keras.optimizers.AdamW"]
  tuner:
    function: "keras_tuner.BayesianOptimization"
    parameters:
      objective: "val_los"
      max_trials: 10
      directory: "tuner_dir"
      project_name: "HPTuning"
    
  sequence:
    - generic_tuning
```

### logging

Controls the logging verbosity.

- `level`: `'INFO'`, `'DEBUG'`, `'ERROR'`, etc.

```yaml
logging:
  level: INFO
```


## Section Descriptions










### `defaults`

Hydra syntax for structured composition. Use `_self_` to preserve the current config after override.

## 💡 Notes

- You can split your config into multiple YAMLs and compose them via the `defaults:` block.
- Use environment variables with `${env:...}` for portable paths.
- For custom experiments, you can easily override any key from the command line:

```bash
tfxkit training.epochs=10 optimizer.learning_rate=0.0005
```