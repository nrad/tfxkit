TFxKit
======

TensorFlow Utilities for NeuralNet Modeling Workflows

TFxKit is a modular Python package that provides a collection of tools, components, and utilities for building, training, and evaluating machine learning models.

It is designed with:

- **Hydra/OmegaConf** for flexible configuration management  
- **Keras/TensorFlow** for model building  
- **Scikit-learn & Pandas** for data preprocessing and tabular input support  
- **Hyperparameter tuning** with Keras Tuner  
- **Clear modular separation** of components for extensibility and reuse  

----

Key Features
------------

- **ModelFactory**: Orchestrates end-to-end ML workflows: data loading, model building, training, evaluation, tuning.  
- **DataManager**: Handles file I/O, feature-label extraction, and sample weights.  
- **ModelBuilder**: Dynamically creates MLP (default) and other user defined models with configurable layers, dropout, batch norm, and more.  
- **Evaluator**: Computes evaluation metrics and attaches model predictions to test/train sets.  
- **HyperTuner**: Wraps KerasTuner with domain-specific enhancements.  
- **Hydra CLI**: Enables powerful command-line execution with dynamic overrides.  

----