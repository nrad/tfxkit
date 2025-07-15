import tensorflow as tf
import logging
import keras


logger = logging.getLogger(__name__)

def xy_maker(
    df, 
    features, 
    labels ):
    """
    Simple xy_maker function that extracts features and labels from a DataFrame."""
    X = df[features]
    y = df[labels]
    return X, y

def define_mlp(
    n_features,
    layers_list,
    hidden_activation="relu",
    n_labels=1,
    dropout=None,
    batch_norm_features=True,
    batch_norm_hidden=True,
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
        n_features (int): Number of input features.
        layers_list (list of int): Number of units for each hidden layer.
        hidden_activation (str or list): Activation(s). Single string or list of same length as layers_list.
        n_labels (int): Number of output units (e.g., for classification).
        dropout (float or list/None): Dropout rate(s). Single or list matching layers_list length.
        batch_norm_features (bool or list of bool): Whether to use BatchNorm for the input layers.
        batch_norm_hidden (bool or list of bool): Whether to use BatchNorm for hidden layers. Single or list of same length as layers_list.
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
        batch_norm_hidden, n_layers, default=False, fill_empty=True
    )
    init_list = process_argument(kernel_initializer, n_layers, default=None)
    reg_list = process_argument(kernel_regularizer, n_layers, default=None)

    sequence = []

    if batch_norm_features:
        sequence.append(keras.layers.BatchNormalization(name="BatchNorm_Input"))
        
    for i in range(n_layers):
        # Optionally add BatchNormalization for this layer

        # Add Dense layer with corresponding kernel init/reg
        dense_kwargs = {
            "activation": activations_list[i] if not batch_norm_all else None,
            "kernel_initializer": init_list[i],
            "kernel_regularizer": parse_regularizer(reg_list[i]),
            "name": f"Dense_{i}_{layers_list[i]}_{activations_list[i]}",
        }
        sequence.append(keras.layers.Dense(layers_list[i], **dense_kwargs))
        if batch_norm_list[i]:
            sequence.append(keras.layers.BatchNormalization(name=f"BatchNorm_{i}"))



        # Optionally add BatchNormalization after this layer
        if batch_norm_all:
            sequence.append(keras.layers.BatchNormalization(name=f"BatchNorm_{i+1}"))
            sequence.append(keras.layers.Activation(activations_list[i]))
            # assert False

        # Optionally add Dropout
        if dropout_list[i]:
            rate = dropout_list[i]
            sequence.append(keras.layers.Dropout(rate, name=f"Dropout_{i}_{rate}"))

    # Add final output layer
    if n_labels:
        sequence.append(
            keras.layers.Dense(n_labels, activation=final_activation, name="Output_Layer")
        )

    if sequence_only:
        return sequence

    # Build the model
    model = tf.keras.Sequential(sequence, name=name)
    if build:
        model.build(input_shape=(batch_size, n_features))
    return model

##
## Utilities for parsing and processing arguments
##

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

