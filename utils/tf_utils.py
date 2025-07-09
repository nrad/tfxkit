# from utils import *

import utils
import tensorflow as tf

# from tensorflow import keras
import keras
from keras import layers


from sklearn.metrics import confusion_matrix
import sklearn
import numpy as np
import pandas as pd
import warnings
import pylab as plt
import os


def add_balanced_weights(
    df,
    label="Level3",
    weight_col="weights",
    balanced_weight_col="balanced_weights",
    factor=1e6,
    weight_func=None,
):
    """
    Add weights based on the class <label> for balancing test/train in the dataframe

    """
    cls0 = df[label] == 0
    cls1 = df[label] == 1

    weights_cls0 = utils.combine_weight_columns(df[cls0], weight_col)
    weights_cls1 = utils.combine_weight_columns(df[cls1], weight_col)

    if weight_func:
        if not callable(weight_func):
            raise ValueError(f"weight_func ({weight_func}) should be callable")
        weights_cls0 = weight_func(weights_cls0)
        weights_cls1 = weight_func(weights_cls1)

    factor /= weights_cls0.sum()

    df.loc[cls1, balanced_weight_col] = (
        weights_cls1 * weights_cls0.sum() / weights_cls1.sum()
    ) * factor
    df.loc[cls0, balanced_weight_col] = weights_cls0 * factor

    return df


def get_class_weight(y, inverse=False, opt="balanced"):
    """
    Compute class weights for imbalanced datasets

    Parameters:
        y (array-like): The target variable.
        inverse (bool, optional): Whether to compute inverse class weights. Defaults to False.
        opt (str, optional): The method to compute class weights. Defaults to "balanced".

    Returns:
        dict: A dictionary containing the class weights for each class.
    """
    weights = sklearn.utils.class_weight.compute_class_weight(
        opt,
        classes=np.unique(y),
        y=y.flatten() if hasattr(y, "flatten") else y.to_numpy().flatten(),
        # y=np.squeeze(y, axis=1)
    )
    if inverse:
        weights = 1.0 / weights

    class_weight = dict(enumerate(weights))
    return class_weight


def get_sample_weight(y, class_weight=None):
    """
    Calculates the sample weights for a binary classification problem.

    Args:
        y (numpy.ndarray): The binary labels of the samples.
        class_weight (dict, optional): The weight assigned to each class. If not provided, it will be calculated using get_class_weight.

    Returns:
        numpy.ndarray: The sample weights.

    """
    class_weight = class_weight if class_weight != None else get_class_weight(y)
    return np.where(y, class_weight[1], class_weight[0])


def define_model(
    n_variables,
    n_labels=3,
    init_layer=64,
    final_layer=4,
    unit_step=2,
    layer_repeat=1,
    max_layers=20,
    name=None,
    hidden_activation="relu",
    dropout=0.8,
    batch_norm=True,
    kernel_regularizer=None,
    kernel_initializer=None,
    # l2_bias_reguali
    batch_size=None,
    batch_norm_all=False,
):
    sequence = []
    if batch_norm:
        pass
        # sequence.append( layers.BatchNormalization(input_shape=(n_variables,)) )
        # assert False
        sequence.append(layers.BatchNormalization())

    n_unit = init_layer
    hid_layers = 0
    hidden_kwargs = dict(
        kernel_regularizer=(
            None
            if kernel_regularizer is None
            else tf.keras.regularizers.l2(kernel_regularizer)
        ),
        kernel_initializer=kernel_initializer,
        activation=hidden_activation,
    )
    while n_unit > final_layer and hid_layers < max_layers:
        for i in range(layer_repeat):
            sequence.append(
                layers.Dense(
                    n_unit,
                    name=f"Dense_{hid_layers}_{n_unit}_{hidden_activation}",
                    **hidden_kwargs,
                )
            )
            if batch_norm_all:
                # assert False
                sequence.append(
                    layers.BatchNormalization(name=f"BatchNorm_{hid_layers}")
                )
                sequence.append(
                    layers.Activation(
                        hidden_activation, name=f"{hidden_activation}_{hid_layers}"
                    )
                )

            if hid_layers == 0 and dropout:
                sequence.append(layers.Dropout(dropout))
            hid_layers += 1
            print(hid_layers, f"Dense_{hid_layers}_{n_unit}_{hidden_activation}")
            if hid_layers >= max_layers:
                break

        n_unit = int(n_unit / unit_step)

    if final_layer:
        if final_layer:
            sequence.append(
                layers.Dense(
                    final_layer,
                    name=f"Dense_{hid_layers}_{final_layer}_{hidden_activation}",
                    **hidden_kwargs,
                ),
            )
    sequence.append(layers.Dense(n_labels, activation="sigmoid"))
    model = tf.keras.Sequential(sequence)
    model.build(input_shape=(batch_size, n_variables))
    return model


def process_argument(arg, n_layers, default=None, fill_empty=True):
    """
    Converts 'arg' into a list of length 'n_layers'.
    If 'arg' is already a list/tuple, validate its length.
    Otherwise, broadcast the single value to all layers.
    """
    if isinstance(arg, (list, tuple)):
        if len(arg) != n_layers:
            raise ValueError(
                f"Length of argument {arg} must match the number of layers ({n_layers})."
            )
        return list(arg)
    else:
        if fill_empty:
            return [arg if arg is not None else default] * n_layers
        else:
            return [arg if arg is not None else default] + [None] * (n_layers - 1)


def parse_regularizer(reg_value):
    """
    Interprets 'reg_value' as:
      - None: no regularization
      - float/int: L2 regularizer with given factor
      - A valid Keras Regularizer instance
    """
    if reg_value is None:
        return None
    if isinstance(reg_value, (float, int)):
        return tf.keras.regularizers.l2(reg_value)
    if isinstance(reg_value, tf.keras.regularizers.Regularizer):
        return reg_value
    raise ValueError(f"Invalid regularizer type: {type(reg_value)}")


def define_flexible_model(
    n_variables,
    layers_list,
    hidden_activation="relu",
    n_labels=3,
    dropout=None,
    batch_norm=False,
    kernel_initializer=None,
    kernel_regularizer=None,
    final_activation="sigmoid",
    name=None,
    build=True,
    batch_size=None,
    sequence_only=False,
    batch_norm_all=False,
):
    """
    Flexible function to define a neural network model with customizable
    layer sizes, hidden_activation, batch norm, dropout, and kernel options.

    Args:
        n_variables (int): Number of input features.
        layers_list (list of int): Number of units for each hidden layer.
        hidden_activation (str or list): Activation(s). Single string or list of same length as layers_list.
        n_labels (int): Number of output units (e.g., for classification).
        dropout (float or list/None): Dropout rate(s). Single or list matching layers_list length.
        batch_norm (bool or list of bool): Whether to use BatchNorm. Single or list matching layers_list length.
                                           If the first element/flag is True, BatchNorm is placed *before* all Dense layers.
        kernel_initializer (str or list/None): Initializer name(s) for each layer. Single or list.
        kernel_regularizer (float/int or list/None): Regularizer factor(s) or Keras Regularizer(s). Single or list.
        name (str or None): Optional name for the model.

    Returns:
        tf.keras.Sequential: A compiled (but not trained) Keras Sequential model.
    """
    layers_list = (
        layers_list if isinstance(layers_list, (list, tuple)) else [layers_list]
    )
    n_layers = len(layers_list)

    # Convert single args to lists or validate list lengths
    activations_list = process_argument(hidden_activation, n_layers, default="relu")
    dropout_list = process_argument(dropout, n_layers, default=None, fill_empty=False)
    print(dropout_list)
    batch_norm_list = process_argument(
        batch_norm, n_layers, default=False, fill_empty=False
    )
    init_list = process_argument(kernel_initializer, n_layers, default=None)
    reg_list = process_argument(kernel_regularizer, n_layers, default=None)

    sequence = []

    # # If first layer's batch_norm is True, apply BatchNorm before any Dense layer
    # if batch_norm_list[0] is True:
    #     sequence.append(layers.BatchNormalization(name="BatchNorm_Input"))
    #     batch_norm_list[0] = False  # Prevent duplication below

    # Build the hidden layers
    for i in range(n_layers):
        # Optionally add BatchNormalization for this layer
        if batch_norm_list[i]:
            sequence.append(layers.BatchNormalization(name=f"BatchNorm_{i}"))

        # Add Dense layer with corresponding kernel init/reg
        dense_kwargs = {
            "activation": activations_list[i] if not batch_norm_all else None,
            "kernel_initializer": init_list[i],
            "kernel_regularizer": parse_regularizer(reg_list[i]),
            "name": f"Dense_{i}_{layers_list[i]}_{activations_list[i]}",
        }
        sequence.append(layers.Dense(layers_list[i], **dense_kwargs))

        # Optionally add BatchNormalization after this layer
        if batch_norm_all:
            sequence.append(layers.BatchNormalization(name=f"BatchNorm_{i+1}"))
            sequence.append(layers.Activation(activations_list[i]))
            # assert False

        # Optionally add Dropout
        if dropout_list[i]:
            rate = dropout_list[i]
            sequence.append(layers.Dropout(rate, name=f"Dropout_{i}_{rate}"))

    # Add final output layer
    if n_labels:
        sequence.append(
            layers.Dense(n_labels, activation=final_activation, name="Output_Layer")
        )

    if sequence_only:
        return sequence

    # Build the model
    model = tf.keras.Sequential(sequence, name=name)
    if build:
        model.build(input_shape=(batch_size, n_variables))
    return model


def define_transformer_model(
    n_variables,
    layers_list,
    hidden_activation="relu",
    n_labels=3,
    dropout=None,
    batch_norm=False,
    kernel_initializer=None,
    kernel_regularizer=None,
    final_activation="sigmoid",
    name=None,
    build=True,
    batch_size=None,
):

    sequence = define_flexible_model(
        n_variables,
        layers_list,
        hidden_activation=hidden_activation,
        n_labels=n_labels,
        dropout=dropout,
        batch_norm=batch_norm,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        final_activation=final_activation,
        name=None,
        build=False,
        batch_size=batch_size,
        sequence_only=True,
    )

    sequence = []
    for ilayer, n_units in enumerate(layers_list):
        layer = layers.Dense(
            n_units,
            activation=hidden_activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=f"Dense_{ilayer}_{n_units}_{hidden_activation}",
        )
        norm_layer = layers.LayerNormalization()(x)
        sequence.extend([layer, norm_layer])
        if dropout:
            sequence.append(layers.Dropout(dropout))

    x = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Prepare data
    X_expanded = X[..., None]  # Add channel dimension

    # Train
    model.fit(X_expanded, y, batch_size=64, epochs=10)


from utils.muon_embedding import define_muemb_model

##
##
##


def fix_df_hist(df_hist):
    # print(df_hist.columns)
    for col in df_hist.columns:
        # print(col, df_hist[col].dtype==object)
        if df_hist[col].dtype == object:
            expanded_col = pd.DataFrame(df_hist[col].tolist())
            expanded_col.columns = [f"{col}_{i}" for i in expanded_col.columns]
            df_hist.drop(col, axis=1, inplace=True)
            df_hist = pd.concat([df_hist, expanded_col], axis=1)
            # print(list(df_hist.columns))
    return df_hist


def plot_history(history, ylim=(0, 1), xlabel="Epoch", ylabel="", plot_kwargs={}):
    history = getattr(history, "history", history)
    df_hist = pd.DataFrame(history)
    df_hist = fix_df_hist(df_hist)

    val_cols = [col for col in df_hist.columns if "val" in col]
    cols = [c for c in df_hist.columns if c not in val_cols]
    fig, ax = plt.subplots()

    df_hist[cols].plot(xlabel=xlabel, ylabel=ylabel, ylim=ylim, ax=ax, **plot_kwargs)
    if val_cols:
        plt.gca().set_prop_cycle(None)
        df_hist[val_cols].plot(style="--", ax=ax, **plot_kwargs)
        ax.legend(ncol=2, loc="upper right")
    return fig, ax, df_hist


def label_conf_matrix(conf_matrix):
    return dict(
        true_positive=conf_matrix[1][1],
        true_negative=conf_matrix[0][0],
        false_positive=conf_matrix[0][1],
        false_negative=conf_matrix[1][0],
    )


def roc(conf_matrix):
    di = label_conf_matrix(conf_matrix)

    return dict(
        tpr=di["true_positive"] / (di["true_positive"] + di["false_negative"]),
        fpr=di["false_positive"] / (di["false_positive"] + di["true_negative"]),
        tnr=di["true_negative"] / (di["true_negative"] + di["false_positive"]),
        fnr=di["false_negative"] / (di["false_negative"] + di["true_positive"]),
        **di,
    )


def plot_confusion(
    labels,
    pred,
    prob=0.9,
    normalize="true",
    ax=None,
    title=None,
    verbose=True,
    sample_weight=None,
):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    pred_prob = pred > prob
    conf_matrix = confusion_matrix(
        labels, pred_prob, normalize=normalize, sample_weight=sample_weight
    )

    if not ax:
        fig, ax = plt.subplots()

    if normalize in ["true", "pred"]:
        clims = dict(vmin=0, vmax=1)
    ax_ = sns.heatmap(conf_matrix, annot=True, fmt="0.3f", ax=ax, **clims)
    ax.invert_yaxis()
    tile = title if title else "Confusion matrix: p>{:.2f}".format(prob)
    ax.set_title(title)
    ax.set_ylabel("Truth", labelpad=-10)
    ax.set_xlabel("Prediction", labelpad=-10)

    cm = conf_matrix

    n_true = float((labels == 1).sum())
    if verbose:
        print(
            f"""
{normalize = }
number of True: {n_true}
  true positives:     { cm[1][1]:0.2f}%
  false positives:   { cm[1][0]:0.2f}%    
        """
        )
    return ax


def get_model_summary(model, **kwargs):
    import io

    stream = io.StringIO()
    kwargs.setdefault("line_length", 120)
    model.summary(print_fn=lambda x: stream.write(x + "\n"), **kwargs)
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


def set_seeds(seed=999):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)  # sets seeds for base-python, numpy and tf
    # tf.config.experimental.enable_op_determinism()


def dataset_generator(
    file_list,
    chunk_size,
    features,
    labels,
    weights=None,
    type="pandas",
    preproc=True,
    balancer=None,
    where=None,
    balancer_kwargs={},
):
    """
    A wrapper for dataframe_generator

    """

    def gen_wrapper():
        return dataframe_generator(
            file_list,
            chunk_size,
            features,
            labels,
            weights=weights,
            type=type,
            preproc=preproc,
            balancer=balancer,
            where=where,
            balancer_kwargs=balancer_kwargs,
        )

    if not weights:
        dataset = tf.data.Dataset.from_generator(
            gen_wrapper,
            output_signature=(
                tf.TensorSpec(shape=(None, len(features)), dtype=tf.float32),
                tf.TensorSpec(
                    shape=(None, len(labels)), dtype=tf.int32
                ),  # Adjust shape and dtype as per your label data
            ),
        )
    else:
        dataset = tf.data.Dataset.from_generator(
            gen_wrapper,
            output_signature=(
                tf.TensorSpec(shape=(None, len(features)), dtype=tf.float32),
                tf.TensorSpec(shape=(None, len(labels)), dtype=tf.int32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            ),
        )

    return dataset


def dataframe_generator(
    file_list,
    chunk_size,
    features,
    labels,
    weights=None,
    where=None,
    type="pandas",
    preproc=True,
    balancer=None,
    balancer_kwargs={},
):
    """
    creates a generated based on pandas dataframes to be used with `tf.data.Dataset.from_generator`:

    def gen_wrapper():
        return dataframe_generator(fnames, BATCH_SIZE, features, labels)

    train_dataset = tf.data.Dataset.from_generator(
        gen_wrapper,
        output_signature=(
            tf.TensorSpec(shape=(None, len(features)), dtype=tf.float32),
            tf.TensorSpec(shape=(None, len(labels)), dtype=tf.int32)
        )
    )

    """
    n_files = len(file_list)
    columns = features + labels
    if weights:
        columns += [weights]
    columns = utils.unique(columns)

    print(f"reading {n_files} files")
    for ifile, file in enumerate(file_list):
        # print("\nreading file %s/%s" % (ifile + 1, n_files), end="\r")
        # dfs = pd.read_hdf(file, chunksize=batch_size)
        dfs = utils.read_hdf_in_chunks(file, chunksize=chunk_size, where=where)
        for df in dfs:
            if preproc:
                #                df = utils.preproc(df)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    df = utils.preproc(df)

            df = df[columns]
            # df = df.sample(frac=1)[columns]
            if balancer:
                print(f"DEBUG: balancing data using {balancer}, {balancer_kwargs}")
                sampler, df = balance_df(
                    df, features, labels, balancer, **balancer_kwargs
                )
            if type == "pandas":
                if not weights:
                    yield df[features].values, df[labels].values
                else:
                    yield df[features].values, df[labels].values, df[[weights]].values
            elif type in ["tf", "tensorflow"]:
                ds = tf.data.Dataset.from_tensor_slices(
                    (
                        tf.cast(df[features].values, tf.float32),
                        tf.cast(df[labels].values, tf.int32),
                    )
                )
                del df
                yield ds
            else:
                raise ValueError("type should be one of 'pandas' or 'tf'")


def get_imblearn_sampler(method="RandomUnderSampler", **kwargs):
    """
    Returns an imblearn sampler object
    """
    import imblearn

    if not method:
        return None

    if hasattr(imblearn.over_sampling, method):
        sampler = getattr(imblearn.over_sampling, method)(**kwargs)
    elif hasattr(imblearn.under_sampling, method):
        sampler = getattr(imblearn.under_sampling, method)(**kwargs)
    else:
        raise ValueError(f"imblearn has no method {method}")
    return sampler


def balance_df(df, features, labels, method="RandomUnderSampler", **kwargs):
    """
    Balance the dataframe using imblearn methods
    """
    import imblearn

    sampler = get_imblearn_sampler(method=method, **kwargs)
    X = df[features]
    y = df[labels]
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    df_resampled = pd.DataFrame(X_resampled, columns=features)
    df_resampled[labels] = y_resampled
    return sampler, df_resampled


def balanced_dataset_generator(
    file_list,
    chunk_size,
    batch_size,
    features,
    labels,
    balancer=None,
    balancer_kwargs={},
    where=None,
    random_state=None,
):
    """
    A wrapper for BalancedBatchGenerator
    """

    import imblearn
    from imblearn.keras import BalancedBatchGenerator

    columns = features + labels
    dfs = utils.read_hdf_in_chunks(
        file_list, chunksize=chunk_size, where=where, columns=columns
    )
    sampler = get_imblearn_sampler(method=balancer, **balancer_kwargs)
    n_files = len(file_list)
    for df in dfs:
        training_generator = BalancedBatchGenerator(
            df[features],
            df[labels],
            sampler=sampler,
            batch_size=batch_size,
            random_state=random_state,
        )
        for X_batch, y_batch in training_generator:
            yield X_batch, y_batch


def prep_train_test(
    df,
    features,
    labels,
    test_size=0.2,
    df_test=None,
    scaler=None,
    seed=None,
    balance_training=False,
):
    """
    <df_test> will be used to prepare the test dataset, <test_size> will be ignored.
    """

    from sklearn.preprocessing import StandardScaler

    if scaler:
        X = scaler.fit_transform(df[features])
    else:
        X = df[features]

    n_labels = len(labels)
    y = df[labels]

    if not df_test is None:
        if test_size:
            raise ValueError(
                "either <test_size> or <df_test> should be given, not both!"
            )

        if scaler:
            X_test = scaler.transform(df_test[features])
            # X_train = scaler.trans
        else:
            X_test = df_test[features]

        X_train = X
        y_train = y
        y_test = df_test[labels]

    else:
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

    print(
        f"""
    {X_train.shape = } 
    {y_train.shape = } 
    ---
    {X_test.shape  = } 
    {y_test.shape  = }
    
    positive % (train) : {round((y_train==1).sum()/len(y_train),3)*100}%
    positive % (test)  : {round((y_test==1).sum()/len(y_test),3)*100}%
    """
    )

    if balance_training:
        if len(labels) > 1:
            raise NotImplementedError(
                "Balancing is not implemented for multiple labels. "
            )
        X_train_df = pd.DataFrame(X_train, columns=features)
        X_train_df.index = y_train.index
        X_train_df = pd.concat([X_train_df, y_train], axis=1)

        X_train_df_balanced = utils.get_balanced_df(X_train_df, by=labels[0])
        X_train = X_train_df_balanced[features].to_numpy()
        y_train = X_train_df_balanced[labels]

        print(
            f"""
        -------- after balancing
        {X_train.shape = } 
        {y_train.shape = } 
        ---
        trig % (train) : {round((y_train==1).sum()/len(y_train),3)*100}%
        """
        )

    return X_train, X_test, y_train, y_test


##
## Model Modifiers
##


def remove_final_activation(model):
    from tensorflow.keras.models import clone_model
    from tensorflow.keras.layers import Activation

    """
        based on the underrated answer of @carlos-bermudez in: 
        https://stackoverflow.com/questions/45492318/keras-retrieve-value-of-node-before-activation-function
    """

    final_layer_name = model.layers[-1].name

    def f(layer):
        config = layer.get_config()
        # if not isinstance(layer, Activation) and layer.name in model.output_names:
        if not isinstance(layer, Activation) and layer.name in [final_layer_name]:
            print(f"removing activation in {layer}")
            config.pop("activation", None)
        layer_copy = layer.__class__.from_config(config)
        return layer_copy

    copy_model = clone_model(model, clone_function=f)
    copy_model.build(model.input_shape)
    copy_model.set_weights(model.get_weights())
    return copy_model


def add_hidden_layer(
    old_model,
    new_hidden_units,
    new_output_units=None,
    hidden_activation="relu",
    output_activation="sigmoid",
    hidden_name=None,
    output_name=None,
    freeze_weights=True,
    **hidden_kwargs,
):
    """
    Freezes all layers in 'old_model', removes the last layer,
    and appends a new hidden layer + a new output layer.

    Args:
        old_model (tf.keras.Sequential): The original Sequential model.
        new_hidden_units (int): Number of neurons in the new hidden layer.
        new_output_units (int): Number of neurons in the new output layer.
        hidden_activation (str): Activation function for the new hidden layer.
        output_activation (str): Activation function for the new output layer.
        hidden_name (str): Name of the new hidden layer.
        output_name (str): Name of the new output layer.

    Returns:
        tf.keras.Sequential: Modified model with frozen original layers,
                             a new hidden layer, and a new output layer.
    """
    # 1) Freeze all existing layers
    old_layers = old_model.layers
    dense_layers = [k for k in old_layers if isinstance(k, keras.layers.Dense)]
    print(dense_layers)
    n_hidden_layers = len(dense_layers) - 1
    if freeze_weights:
        for layer in old_layers:
            layer.trainable = False

    # 2) Remove the last layer
    frozen_layers = old_layers[:-1]
    new_model = tf.keras.Sequential(frozen_layers)

    # 3) Add the new hidden layer
    hidden_name = (
        hidden_name
        if hidden_name
        else f"Dense_{n_hidden_layers}_{new_hidden_units}_{hidden_activation}"
    )
    new_model.add(
        layers.Dense(
            new_hidden_units,
            activation=hidden_activation,
            name=hidden_name,
            **hidden_kwargs,
        )
    )

    # 4) Add the new output layer
    output_name = output_name if output_name else f"Dense_output_{output_activation}"
    new_model.add(
        layers.Dense(
            new_output_units if new_output_units else old_layers[-1].units,
            activation=output_activation,
            name=output_name,
        )
    )
    # n_variables
    new_model.build(old_model.inputs[0].shape)
    return new_model


def get_dense_layers(model, hidden_only=False):
    """
    Returns all Dense layers in a model.
    """
    dense_layers = [k for k in model.layers if isinstance(k, keras.layers.Dense)]
    if hidden_only:
        return dense_layers[:-1]
    return dense_layers


##
##
##


def get_feature_permutation_importance(
    model, X, y, key=None, features=None, N=10_000, n_repeats=10
):
    """
    evaluate the model after permutating the values in each feature.
    Importance is measured as the change in a given metric.
    """
    from sklearn.inspection import permutation_importance

    features = features if features else X.columns

    def keras_model_score(model, X, y):
        di = model.evaluate(X, y, verbose=0, return_dict=True)
        # di['neg_loss'] = -di.pop('loss')
        return di

    perm_importance = permutation_importance(
        estimator=model,
        X=X.sample(N) if N else X,
        y=y.sample(N) if N else y,
        scoring=lambda estimator, X, y: keras_model_score(estimator, X, y),
        n_repeats=n_repeats,
    )
    keys = [key] if key else list(perm_importance.keys())
    # fig, ax = plt.subplots(n
    ret = {}
    for ikey, key in enumerate(keys):
        # fig, ax =
        # ax = axs[ikey]
        fig, ax = plt.subplots()
        sorted_idx = perm_importance[key].importances_mean.argsort()
        ax.barh(
            np.array(features)[sorted_idx],
            perm_importance[key].importances_mean[sorted_idx],
            alpha=0.8,
            height=0.8,
        )
        ax.errorbar(
            perm_importance[key].importances_mean[sorted_idx],
            range(len(features)),
            xerr=2 * perm_importance[key].importances_std[sorted_idx],
            color="C1",
            fmt="o",
            markersize=1,
        )  # linewidth=10, alpha=0.5)
        ax.set_xlabel("Permutation Importance (%s)" % key)
        ret[key] = {
            "fig": fig,
            "ax": ax,
            "sorted_features": np.array(features)[sorted_idx],
        }
    return ret


def make_epoch_checkpoint_callback(
    filedir="",
    filename="model_checkpoint_epoch_{epoch:02d}.weights.h5",
    save_best_only=False,
    save_weights_only=True,
    save_freq="epoch",
    verbose=1,
    **kwargs,
):
    filepath = os.path.join(filedir, filename)
    print("callback dir: %s" % filepath)
    epoch_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        save_best_only=save_best_only,
        save_weights_only=save_weights_only,
        save_freq=save_freq,
        verbose=verbose,
        **kwargs,
    )
    return epoch_checkpoint_callback


###
### Keras custom LR schedules
###


lr_schedule_exponential_decay = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9,
    name="ExponentialDecay",
)

lr_schedule_inverse = keras.optimizers.schedules.InverseTimeDecay(
    1e-2,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=False,
    name="InverseTimeDecay",
)

lr_dict = {
    "0.01": 1e-2,
    "0.001": 1e-3,
    "0.0001": 1e-4,
    "decay_exp": lr_schedule_exponential_decay,
    "decay_linear": lr_schedule_inverse,
}


def get_learning_rate(lr):
    if lr in lr_dict:
        return lr_dict[lr]
    elif isinstance(lr, str):
        try:
            return float(lr)
        except Exception as e:
            print(f"Learning rate ({lr}) could not be converted to float!")
            print(e)
            raise e
    elif isinstance(lr, float):
        return lr
    else:
        raise ValueError(
            f"Not sure how to deal with learning rate: {lr} of type: <{type(lr)}>."
        )


def get_clipnorm(clipnorm):
    if clipnorm == 0:
        return None
    return clipnorm


#####
##### TF Custom Metric
#####


def cumilative_prediction_gap(y_true, y_pred):
    """
    Calculate the metric based on the difference in areas under the cumulative
    distributions of predicted probabilities for positive and negative classes.

    This metric computes separate histograms for the predictions corresponding
    to actual positive and negative outcomes. It then calculates the cumulative
    sum (normalized by the total count in each class) for these histograms,
    representing the cumulative distribution of predictions for each class.
    The area under each cumulative distribution curve is computed, and the
    metric value is the difference between these two areas. The metric value
    will range between 0 and 1.


    Parameters:
    y_true (Tensor): True labels. Expected to be binary (0 or 1).
    y_pred (Tensor): Predicted probabilities. Expected to be in the range [0, 1].

    Returns:
    float: The computed metric value, representing the difference in areas
    under the cumulative distribution curves of the positive and negative classes.

    Note:
    This function is designed to work with batched data, and the metric is
    computed for each batch independently. Care should be taken when interpreting
    this metric across different batch sizes.
    """  # Convert to numpy arrays for histogram computation
    y_true_np = y_true
    y_pred_np = y_pred

    # print(y_true_np, y_pred_np)
    # Separate the predictions into positive and negative classes
    pos_predictions = y_pred_np[y_true_np == 1]
    neg_predictions = y_pred_np[y_true_np == 0]

    if not len(pos_predictions) or not len(neg_predictions):
        return 0

    # Compute histograms
    pos_hist, _ = np.histogram(pos_predictions, bins=100, range=(0, 1), density=True)
    neg_hist, _ = np.histogram(neg_predictions, bins=100, range=(0, 1), density=True)

    assert not np.isnan(pos_hist).any(), (
        "NaNs found in pos_hist",
        pos_hist,
        pos_predictions,
    )
    assert not np.isnan(neg_hist).any(), (
        "NaNs found in neg_hist",
        neg_hist,
        neg_predictions,
    )

    # Compute cumulative fractions
    pos_cum_frac = np.cumsum(pos_hist) / np.sum(pos_hist)
    neg_cum_frac = np.cumsum(neg_hist) / np.sum(neg_hist)

    # Compute areas (integral of the cumulative fractions)
    area_pos = np.trapz(pos_cum_frac, dx=1 / len(pos_cum_frac))
    area_neg = np.trapz(neg_cum_frac, dx=1 / len(neg_cum_frac))

    # Calculate the metric
    metric_value = area_neg - area_pos

    # print(y_true_np.shape, metric_value)
    return metric_value


@tf.function
def CPG(y_true, y_pred):
    metric_value = tf.py_function(
        cumilative_prediction_gap, [y_true, y_pred], Tout=tf.float32
    )
    return metric_value


@keras.saving.register_keras_serializable(
    package="MyMetrics", name="CumilativePredictionGap"
)
class CumilativePredictionGap(tf.keras.metrics.Metric):
    """
    see cumilative_prediction_gap

    """

    def __init__(self, name="CPG", **kwargs):
        super().__init__(name=name, **kwargs)
        # Additional initialization can go here if necessary

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Assuming y_true is binary with {0, 1} values
        y_true = tf.cast(y_true, tf.bool)
        positive_preds = tf.boolean_mask(y_pred, y_true)
        negative_preds = tf.boolean_mask(y_pred, ~y_true)

        # Compute histograms - this is a simplified example
        print(f"{positive_preds=}")
        positive_hist = tf.histogram_fixed_width(positive_preds, [0, 1], nbins=100)
        negative_hist = tf.histogram_fixed_width(negative_preds, [0, 1], nbins=100)
        print(f"{positive_hist=}")
        # Compute cumulative distributions
        positive_cdf = tf.cumsum(positive_hist) / tf.reduce_sum(positive_hist)
        negative_cdf = tf.cumsum(negative_hist) / tf.reduce_sum(negative_hist)

        # Store cumulative distributions as part of the metric state
        # In practice, you might need to handle batch-wise accumulation and averaging
        self.positive_cdf = positive_cdf
        self.negative_cdf = negative_cdf

    def result(self):
        # Compute the area (integral) under the cumulative distribution curves
        # Here we use a simple trapezoidal rule for demonstration
        area_positive = tf.reduce_sum(self.positive_cdf) / tf.cast(
            tf.size(self.positive_cdf), tf.float64
        )
        area_negative = tf.reduce_sum(self.negative_cdf) / tf.cast(
            tf.size(self.negative_cdf), tf.float64
        )

        # Return the difference in areas
        return area_negative - area_positive

    def reset_state(self):
        # Reset the state of the metric, to be ready for the next epoch
        pass  # Add reset logic if your metric maintains a state across batches

    def get_config(self):
        # Return the configuration of the metric
        config = super().get_config()
        # Add any other arguments to config if necessary
        return config
