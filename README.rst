TFxKit
======

TensorFlow Utilities for NeuralNet Modeling Workflows

TFxKit is a modular Python package that provides a collection of tools, components, and utilities for building, training, and evaluating machine learning models.

It is designed with:

- **Hydra/OmegaConf** for flexible configuration management  
- **Keras/TensorFlow** for model building  
- **Scikit-learn & Pandas** for data preprocessing and tabular input support  
- **Hyperparameter tuning** with Keras Tuner  
- **Visualization** with Matplotlib and Seaborn, plothist, shapely, etc.
- **Clear modular separation** of components for extensibility and reuse  
- **Comprehensive logging** for easy debugging and monitoring

----

Key Features
------------

- **ModelFactory**: Orchestrates end-to-end ML workflows: data loading, model building, training, evaluation, tuning.
    
    .. code-block:: python

        from tfxkit import ModelFactory
        mf = ModelFactory("/path/to/config.yaml")
        mf.fit()
        mf.attach_predictions()
        mf.make_plots()
        mf.hyper_tune()

- **DataManager**: Handles file I/O, feature-label extraction, and sample weights.  

    .. code-block:: python

        mf.test.X # to access the test features
        mf.train.sample_weight # to access training sample weights

- **ModelBuilder**: Dynamically creates MLP (default) and other user defined models with configurable layers, dropout, batch norm, and more.  

    .. code-block:: python

        model = mf.builder.define_model()
        mf.summary()

- **Trainer**: Manages model training with support for early stopping, learning rate scheduling, and custom callbacks.    

    .. code-block:: python

        history = mf.fit() 
        mf.plot_history()

- **Evaluator**: Computes evaluation metrics and attaches model predictions to test/train sets.  
- **HyperTuner**: Wraps KerasTuner with domain-specific enhancements.  

    .. code-block:: python

        tuner = mf.hyper_tuner()
        tuner.search(mf.train.X, mf.train.y, ...)

- **Hydra CLI**: Enables command-line execution, dynamically overriding config parameters.

    .. code-block:: bash

        tfxkit --config-path /path/to/config.yaml trainer.epochs=50 model.parameters.dropout=0.3 


----