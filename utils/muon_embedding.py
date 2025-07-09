import tensorflow as tf

import keras

# from keras import layers, Model, Input, Sequential
import re
import numpy as np
import pandas as pd


# muon_features = ["pos_x", "pos_y", "pos_z", "dir_x", "dir_y", "dir_z", "radius", "energy"]
DEFAULT_MUON_FEATURES = [
    "pos_x",
    "pos_y",
    "pos_z",
    "dir_x",
    "dir_y",
    "dir_z",
    "radius",
    "log_energy",
]
DEFAULT_N_MUONS = 10
# n_muon_features = len(muon_features)


def df_to_muon_array(
    df, n_muons=10, epsilon_std=1e-5, muon_features=DEFAULT_MUON_FEATURES
):
    """
    Converts columns like mu1_pos_x, mu1_pos_y, mu1_pos_z, mu1_dir_x, mu1_dir_y, mu1_dir_z, mu1_energy
    (and similarly for mu2..mu10) into a NumPy array of shape (num_events, n_muons, 7).

    Each "slice" along axis 1 corresponds to one muon, with the 7 features:
      [pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, energy].

    If muon feature values are zero, replaces them with small random numbers sampled
    from a normal distribution around zero with standard deviation `epsilon_std`.
    """
    arr = np.zeros((len(df), n_muons, len(muon_features)), dtype=float)

    for i in range(1, n_muons + 1):
        col_list = [f"mu{i}_{feat}" for feat in muon_features]
        print(f"muon {i}: {col_list}")
        muon_data = df[col_list].values
        if epsilon_std is not None:
            zero_mask = muon_data == 0
            muon_data[zero_mask] = np.random.normal(
                loc=0, scale=epsilon_std, size=zero_mask.sum()
            )
        else:
            zero_mask = muon_data == 0
            muon_data[zero_mask] = np.nan

        arr[:, i - 1, :] = muon_data
    return arr


def df_to_muon_plus_event_array(
    df,
    n_muons=10,
    epsilon_std=1e-5,
    muon_features=DEFAULT_MUON_FEATURES,
    event_features=[],
):
    """
    similar to df_to_muon_array but also includes the event information per muon
    """
    # event_features = [k for k in df.columns if not re.findall(r"mu\d+_*", k)]
    print(event_features)
    arr = np.zeros(
        (len(df), n_muons, len(muon_features) + len(event_features)), dtype=float
    )

    for i in range(1, n_muons + 1):
        col_list = [f"mu{i}_{feat}" for feat in muon_features] + event_features
        print(f"muon {i}: {col_list}")
        muon_data = df[col_list].values
        if epsilon_std is not None:
            zero_mask = muon_data == 0
            muon_data[zero_mask] = np.random.normal(
                loc=0, scale=epsilon_std, size=zero_mask.sum()
            )
        else:
            zero_mask = muon_data == 0
            muon_data[zero_mask] = np.nan

        arr[:, i - 1, :] = muon_data
    return arr


def df_to_muon_ragged_array(df, n_muons=DEFAULT_N_MUONS):
    """
    Converts a DataFrame with columns like mu1_pos_x, mu1_pos_y, ..., mu10_energy
    into a ragged tensor of shape (num_events, None, 7).

    Each row corresponds to an event, and each inner row corresponds to a muon with its 7 features:
      [pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, energy].

    Args:
        df (pd.DataFrame): Input DataFrame with mu1_pos_x...mu10_energy columns.
        n_muons (int): Maximum number of muons (column pattern).

    Returns:
        tf.RaggedTensor: Ragged tensor of shape (num_events, None, 7).
    """
    ragged_data = []

    for _, row in df.iterrows():
        muon_data = []
        for i in range(1, n_muons + 1):
            # Extract features for each muon
            muon_features = [row.get(f"mu{i}_{feat}", None) for feat in muon_features]
            # Only append if all features are non-NaN (muon exists)
            if all(f != 0 for f in muon_features):
                muon_data.append(muon_features)
        ragged_data.append(muon_data)

    # Convert to TensorFlow RaggedTensor
    ragged_tensor = tf.ragged.constant(ragged_data, dtype=tf.float32)
    raise NotImplementedError("This function is not fully implemented yet")
    return ragged_tensor


def build_muon_branch(
    embedding_dim=16,
    layers_list=[1028, 512, 256, 128, 64],
    hidden_activation="relu",
    dropout=0.2,
    kernel_regularizer=1e-4,
    final_activation="sigmoid",
    n_muon_features=len(DEFAULT_MUON_FEATURES),
    n_muons=DEFAULT_N_MUONS,
    aggregation_method="simple",
    batch_size=None,
    **kwargs,
):
    """
    Returns a Sequential sub-model that:
      - Uses TimeDistributed to apply a small MLP to each muon in the bundle.
      - Aggregates embeddings (e.g., by mean, max, etc).
    """
    from utils import tf_utils

    # muon_input = keras.Input(shape=(None, n_muon_features), name="muon_input")
    muon_input = keras.Input(shape=(n_muons, n_muon_features), name="muon_input")
    muon_mlp = tf_utils.define_flexible_model(
        n_muon_features,
        layers_list=layers_list,
        hidden_activation=hidden_activation,
        n_labels=embedding_dim,
        batch_norm=True,
        dropout=dropout,
        kernel_regularizer=kernel_regularizer,
        final_activation=final_activation,
        batch_size=batch_size,
        build=True,
        name="MuonEmbedding",
        **kwargs,
    )
    muon_mlp.summary()

    ##
    ## Apply the same MLP to each muon in the bundle,
    ## i.e. muon_multiplicity is considered as the temporal dimension
    ##
    print(muon_mlp)
    print(type(muon_mlp))
    muon_embeddings = keras.layers.TimeDistributed(muon_mlp)(muon_input)

    # aggregation =
    if not aggregation_method in AGGREGATION_METHODS:
        raise ValueError(
            f"Invalid aggregation method: {aggregation_method}. Must be one of {list(AGGREGATION_METHODS.keys())}"
        )
    agg_layer = AGGREGATION_METHODS[aggregation_method]()
    # if hasattr(agg_layer, "build"):
    #     print(f"building agg_layer with input_shape {muon_embeddings.shape}")
    #     #agg_layer.build(input_shape=muon_embeddings.shape)
    #     agg_layer.build(input_shape=(batch_size, DEFAULT_N_MUONS, n_muon_features) )

    muon_event_embedding = agg_layer(muon_embeddings)

    return keras.Model(
        inputs=muon_input, outputs=muon_event_embedding, name="MuonBranch"
    )


# tf.keras.regularizers.l2(kernel_regularizer)
# mf.features = event_features


def build_event_branch(
    event_feat_dim=10,
    layers_list=[1028, 512, 256, 128, 64],
    hidden_activation="relu",
    n_final_units=64,
    final_activation="sigmoid",
    dropout=0.2,
    kernel_regularizer=1e-4,
    **kwargs,
):
    """
    Returns a Sequential sub-model for the event-level features.
    Example: 2 dense layers, can be adapted as needed.
    """
    from utils import tf_utils

    event_input = keras.Input(shape=(event_feat_dim,), name="event_input")
    event_mlp = tf_utils.define_flexible_model(
        n_variables=event_feat_dim,
        layers_list=layers_list,
        hidden_activation=hidden_activation,
        n_labels=n_final_units,
        batch_norm=True,
        dropout=dropout,
        kernel_regularizer=kernel_regularizer,
        final_activation=final_activation,
        build=True,
        name="EventMLP",
        **kwargs,
    )
    event_mlp(event_input)
    return event_mlp


def build_combined_model(
    event_feat_dim=10,
    muon_embedding_dim=16,
    muon_feat_dim=len(DEFAULT_MUON_FEATURES),
    layers_list=[1028],
    kernel_regularizer=1e-4,
    dropout=0.2,
    event_branch_kwargs={},
    muon_branch_kwargs={},
    combined_activation="relu",
    batch_size=None,
):
    from utils import tf_utils

    print(f"{event_branch_kwargs = }")
    event_branch = build_event_branch(
        event_feat_dim=event_feat_dim, **event_branch_kwargs
    )
    print(f"{muon_branch_kwargs = }")
    layers_list = layers_list if isinstance(layers_list, list) else [layers_list]
    dropout_list = tf_utils.process_argument(dropout, n_layers=len(layers_list))
    muon_branch = build_muon_branch(
        embedding_dim=muon_embedding_dim,
        n_muon_features=muon_feat_dim,
        # batch_size=batch_size,
        **muon_branch_kwargs,
    )
    combined = keras.layers.Concatenate(name="merge_branches")(
        [event_branch.output, muon_branch.output]
    )

    x = keras.layers.BatchNormalization()(combined)
    for i, n_units in enumerate(layers_list):
        if i != 0:
            x = keras.layers.Dropout(dropout_list[i - 1])(x)
        x = keras.layers.Dense(
            n_units,
            activation=combined_activation,
            kernel_regularizer=tf_utils.parse_regularizer(kernel_regularizer),
        )(x)

    output = keras.layers.Dense(1, activation="sigmoid", name="classification")(x)
    model = keras.Model(
        inputs=[event_branch.input, muon_branch.input],
        outputs=output,
        name="CombinedModel",
    )

    return event_branch, muon_branch, model


# event_features = [k for k in mf.features if not (re.findall(r"mu\d+_pos_[xyz]", k) or re.findall(r"mu\d+_dir_[xyz]", k) or re.findall("mu\d+_energy", k) or re.findall("mu\d+_log_energy", k)) ]

# n_event_features = len(event_features)


def xy_maker_muon_embedding(
    mf, df=None, sample_weight=None, muon_features=DEFAULT_MUON_FEATURES
):
    event_features = [k for k in mf.features if not re.findall(r"mu\d+_*", k)]
    df = df if df is not None else mf.df_train
    event_labels = df[mf.labels[0]] == 1
    event_feats = df[event_features]
    muon_feats = df_to_muon_array(df, muon_features=muon_features)
    print("-----------------")
    print("-----------------")
    print(f"{event_features = }")
    print("-----------------")
    print(f"{muon_features = }")
    print("-----------------")
    print("-----------------")
    assert False
    return dict(x=[event_feats, muon_feats], y=event_labels)


def xy_maker_muon_embedding2(
    mf, df=None, sample_weight=None, muon_features=DEFAULT_MUON_FEATURES
):
    event_features = [k for k in mf.features if not re.findall(r"mu\d+_*", k)]
    df = df if df is not None else mf.df_train
    event_labels = df[mf.labels[0]] == 1
    event_feats = df[event_features]
    muon_feats = df_to_muon_array(df, muon_features=muon_features, epsilon_std=0)

    xy = dict(x=[event_feats, muon_feats], y=event_labels)
    if sample_weight is not None:
        if isinstance(sample_weight, str):
            sample_weight = df[sample_weight]
        else:
            sample_weight = sample_weight
        xy["sample_weight"] = sample_weight

    print("-----------------")
    print("-----------------")
    print(f"{event_features = }")
    print("-----------------")
    print(f"{muon_features = }")
    print("-----------------")
    print("-----------------")

    return xy


def xy_maker_muonevent_embedding(mf, df=None, muon_features=DEFAULT_MUON_FEATURES):
    """
    combining event features with each muon
    """
    event_features = [k for k in mf.features if not re.findall(r"mu\d+_*", k)]
    df = df if df is not None else mf.df_train
    event_labels = df[mf.labels[0]] == 1
    event_feats = df[event_features]
    muon_feats = df_to_muon_plus_event_array(
        df, muon_features=muon_features, event_features=event_features, epsilon_std=0
    )

    print("-----------------")
    print("-----------------")
    print(f"{event_features = }")
    print("-----------------")
    print(f"{muon_features = }")
    print("-----------------")
    print("-----------------")

    return dict(x=[event_feats, muon_feats], y=event_labels)


##
## Model Builder
##


def define_muemb_model(
    n_variables,
    n_labels,
    features=[],
    event_branch_layers=[64],
    muon_branch_layers=[64],
    combination_layers=[64],
    muon_embedding_dim=16,
    hidden_activation="relu",
    dropout=0.3,
    dropout_muon=None,
    dropout_event=None,
    kernel_regularizer=1e-4,
    aggregation_method="simple",
    batch_size=None,
    batch_norm_all=False,
):
    event_features = [k for k in features if not re.findall(r"mu\d+_*", k)]
    n_event_features = len(event_features)
    muon_features = [k for k in features if re.findall(r"mu\d+_*", k)]
    print(f"{n_event_features=}, \n{muon_features=} \n{features=}")
    event_branch_kwargs = dict(
        layers_list=event_branch_layers,
        dropout=dropout_event if dropout_event is not None else dropout,
        kernel_regularizer=kernel_regularizer,
        batch_norm_all=batch_norm_all,
    )
    muon_branch_kwargs = dict(
        layers_list=muon_branch_layers,
        dropout=dropout_muon if dropout_muon is not None else dropout,
        kernel_regularizer=kernel_regularizer,
        aggregation_method=aggregation_method,
        batch_size=batch_size,
        batch_norm_all=batch_norm_all,
    )

    print(muon_branch_kwargs)
    event_branch, muon_branch, model = build_combined_model(
        event_feat_dim=n_event_features,
        muon_embedding_dim=muon_embedding_dim,
        layers_list=combination_layers,
        event_branch_kwargs=event_branch_kwargs,
        muon_branch_kwargs=muon_branch_kwargs,
        dropout=dropout,
        batch_size=batch_size,
    )
    model.event_branch = event_branch
    model.muon_branch = muon_branch

    return model


def define_muevtemb_model(
    n_variables,
    n_labels,
    features=[],
    event_branch_layers=[64],
    muon_branch_layers=[64],
    combination_layers=[64],
    muon_embedding_dim=16,
    hidden_activation="relu",
    dropout=0.3,
    dropout_muon=None,
    dropout_event=None,
    kernel_regularizer=1e-4,
    aggregation_method="simple",
    batch_size=None,
    batch_norm_all=False,
):
    """
    Similar to define_muemb_model, but combines event features with each muon."""

    event_features = [k for k in features if not re.findall(r"mu\d+_*", k)]
    n_event_features = len(event_features)
    muon_features = [k for k in features if re.findall(r"mu\d+_*", k)]
    print(f"{n_event_features=}, \n{muon_features=} \n{features=}")
    event_branch_kwargs = dict(
        layers_list=event_branch_layers,
        dropout=dropout_event if dropout_event is not None else dropout,
        kernel_regularizer=kernel_regularizer,
        batch_norm_all=batch_norm_all,
    )
    muon_branch_kwargs = dict(
        layers_list=muon_branch_layers,
        dropout=dropout_muon if dropout_muon is not None else dropout,
        kernel_regularizer=kernel_regularizer,
        aggregation_method=aggregation_method,
        batch_size=batch_size,
        batch_norm_all=batch_norm_all,
    )

    print(muon_branch_kwargs)
    print(muon_features)
    print("event", event_features)
    # assert False, len(muon_features) + n_event_features

    event_branch, muon_branch, model = build_combined_model(
        event_feat_dim=n_event_features,
        muon_embedding_dim=muon_embedding_dim,
        muon_feat_dim=len(DEFAULT_MUON_FEATURES) + n_event_features,
        layers_list=combination_layers,
        event_branch_kwargs=event_branch_kwargs,
        muon_branch_kwargs=muon_branch_kwargs,
        dropout=dropout,
        batch_size=batch_size,
    )
    model.event_branch = event_branch
    model.muon_branch = muon_branch

    return model


# Example usage:
#  - Suppose event_feat_dim=10 (10 global features).
#  - Each event has a variable # of muons (n_muons), each muon has 7 features.
# Build and compile:


# mf.X_train

# class WeightedSumAggregation(keras.layers.Layer):
#     def __init__(self, embedding_dim, n_muons=None, **kwargs):
#         super().__init__(**kwargs)
#         self.embedding_dim = embedding_dim
#         self.n_muons = n_muons
#         # self.weights = None  # Placeholder for the learnable weights

#     def build(self, input_shape):
#         # Initialize weights when input shape is fully defined
#         #n_muons = input_shape[1]  # Number of muons (second dimension)
#         n_muons = self.n_muons
#         print(f"WeightedSumAggregation: n_muons={n_muons}: {input_shape=}")
#         if n_muons is None:
#             raise ValueError("Number of muons (n_muons) must be fixed or determined at runtime.")
#         self.weights = self.add_weight(
#             shape=(n_muons, 1),  # One weight per muon
#             initializer="uniform",
#             trainable=True,
#             name="aggregation_weights"
#         )

#     def call(self, inputs):
#         # Normalize weights using softmax
#         weights = tf.nn.softmax(self.weights, axis=0)
#         # Perform weighted sum across muons
#         weighted_sum = tf.reduce_sum(inputs * weights, axis=1)  # Shape: (batch_size, embedding_dim)
#         return weighted_sum

# class AttentionAggregation(keras.layers.Layer):
#     def __init__(self, embedding_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.embedding_dim = embedding_dim

#     def build(self, input_shape):
#         static_shape = input_shape.as_list()
#         # Optionally enforce that the input's embedding dimension matches.
#         if static_shape[-1] is not None and static_shape[-1] != self.embedding_dim:
#             raise ValueError("Input embedding dimension does not match expected embedding_dim.")
#         self.query = self.add_weight(
#             shape=(self.embedding_dim,),
#             initializer="uniform",
#             trainable=True,
#             name="query_vector"
#         )
#         super().build(input_shape)

#     def call(self, inputs):
#         # inputs: (batch_size, n_muons, embedding_dim)
#         # Compute scores via a dot product with the query.
#         scores = tf.reduce_sum(inputs * self.query, axis=-1)  # (batch_size, n_muons)
#         attention_weights = tf.nn.softmax(scores, axis=1)
#         # Use the attention weights to compute a weighted sum.
#         weighted_sum = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, -1), axis=1)
#         return weighted_sum  # (batch_size, embedding_dim)


# class AttentionAggregation(keras.layers.Layer):
#     def __init__(self, embedding_dim, **kwargs):
#         super().__init__(**kwargs)
#         #super(AttentionAggregation, self).__init__(**kwargs)
#         self.embedding_dim = embedding_dim

#     def build(self, input_shape):
#         # Learnable query vector
#         self.query = self.add_weight(
#             shape=(self.embedding_dim,),
#             initializer="uniform",
#             trainable=True,
#             name="query_vector"
#         )

#     def call(self, inputs):
#         # inputs: (batch_size, n_muons, embedding_dim)
#         # Compute attention scores (dot product with query)
#         scores = tf.reduce_sum(inputs * self.query, axis=-1)  # (batch_size, n_muons)
#         attention_weights = tf.nn.softmax(scores, axis=1)     # Normalize weights

#         # Weighted sum of muon embeddings
#         weighted_sum = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, -1), axis=1)
#         return weighted_sum  # (batch_size, embedding_dim)

# class NeuralNetworkAggregation(keras.layers.Layer):
#     def __init__(self, output_dim=16, **kwargs):
#         # super(NeuralNetworkAggregation, self).__init__(**kwargs)
#         super().__init__(**kwargs)
#         self.output_dim = output_dim
#         self.dense1 = keras.layers.Dense(64, activation="relu")
#         self.dense2 = keras.layers.Dense(output_dim, activation="relu")

#     def call(self, inputs):
#         # inputs: (batch_size, n_muons, embedding_dim)
#         # Flatten muons into a single vector per event
#         flattened = tf.reshape(inputs, (tf.shape(inputs)[0], -1))  # (batch_size, n_muons * embedding_dim)
#         print(flattened, flattened.shape)
#         return self.dense2(self.dense1(flattened))  # (batch_size, output_dim)

# class NeuralNetworkAggregation(keras.layers.Layer):
#     def __init__(self, output_dim=16, **kwargs):
#         super().__init__(**kwargs)
#         self.output_dim = output_dim

#     def build(self, input_shape):
#         # input_shape is (batch_size, n_muons, embedding_dim)
#         # Ensure n_muons and embedding_dim are fully defined:
#         # if input_shape[1] is None or input_shape[2] is None:
#         #     raise ValueError("n_muons and embedding_dim must be defined, but got:", input_shape)
#         # n_muons = input_shape[1]
#         # embedding_dim = input_shape[2]
#         # flatten_dim = n_muons * embedding_dim

#         #self.dense1 = keras.layers.Dense(64, activation="relu", input_shape=(flatten_dim,))
#         self.dense1 = keras.layers.Dense(64, activation="relu")
#         self.dense2 = keras.layers.Dense(self.output_dim, activation="relu")
#         super().build(input_shape)

#     # def call(self, inputs):
#     #     # Flatten the muon dimension into a single vector per event
#     #     #flattened = tf.reshape(inputs, (tf.shape(inputs)[0], -1))
#     #     flattened = tf.reshape(inputs, (8192, -1))
#     #     print(flattened, flattened.shape)
#     #     #assert False
#     #     return self.dense2(self.dense1(flattened))
#     def call(self, inputs):
#         # Flatten the muon dimension into a single vector per event,
#         # while keeping the batch dimension dynamic.
#         #flattened = tf.reshape(inputs, (tf.shape(inputs)[0], -1))
#         # flattened = tf.reshape(inputs, (8192, 10*16))
#         # assert False, (inputs, inputs.shape, tf.shape(inputs))

#         flattened = tf.reshape(inputs, (tf.shape(inputs)[0], -1))
#         return self.dense2(self.dense1(flattened))


class NeuralNetworkAggregation(keras.layers.Layer):
    def __init__(self, output_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        # input_shape is expected to be (batch_size, n_muons, embedding_dim)
        static_shape = list(input_shape)
        n_muons = static_shape[1]
        embedding_dim = static_shape[2]
        if n_muons is None or embedding_dim is None:
            raise ValueError("n_muons and embedding_dim must be fully defined.")
        flatten_dim = n_muons * embedding_dim
        # Create Dense layers with the known flattened dimension
        self.dense1 = keras.layers.Dense(
            64, activation="relu", input_shape=(flatten_dim,)
        )
        self.dense2 = keras.layers.Dense(self.output_dim, activation="relu")
        super().build(input_shape)

    def call(self, inputs):
        # Use tf.shape for dynamic batch size, but use static dimensions for flattening.
        batch_size = tf.shape(inputs)[0]
        static_shape = inputs.get_shape().as_list()
        flatten_dim = (
            static_shape[1] * static_shape[2]
        )  # now fully defined, e.g., 10 * 16 = 160
        flattened = tf.reshape(inputs, (batch_size, flatten_dim))
        return self.dense2(self.dense1(flattened))


# class NeuralNetworkAggregation(keras.layers.Layer):
#     def __init__(self, output_dim=16, **kwargs):
#         super().__init__(**kwargs)
#         self.output_dim = output_dim

#     def build(self, input_shape):
#         # Expecting input_shape = (batch_size, n_muons, embedding_dim)
#         _, n_muons, embedding_dim = input_shape
#         if n_muons is None or embedding_dim is None:
#             raise ValueError("Input dimensions n_muons and embedding_dim must be fully defined")
#         flatten_dim = n_muons * embedding_dim
#         self.dense1 = keras.layers.Dense(64, activation="relu", input_dim=flatten_dim)
#         self.dense2 = keras.layers.Dense(self.output_dim, activation="relu")

#     def call(self, inputs):
#         print("====================================================")
#         print("🔍 Received input_shape in:", inputs, tf.shape(inputs))
#         print("🔍 Received input_shape in:", inputs.shape)
#         batch_size = tf.shape(inputs)[0]  # Get dynamic batch size
#         n_muons = tf.shape(inputs)[1]  # Get dynamic number of muons
#         embedding_dim = tf.shape(inputs)[2]  # Should be 16

#         flatten_dim = n_muons * embedding_dim  # Compute dynamically

#         print(f"🛠 call() -> batch: {batch_size}, muons: {n_muons}, embed_dim: {embedding_dim}")


#         flattened = tf.reshape(inputs, (batch_size, flatten_dim))  # Flatten dynamically
#         return self.dense2(self.dense1(flattened))
class WeightedSumAggregation(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agg_weights = None  # This will be created in build()

    def build(self, input_shape):
        # input_shape: (batch_size, n_muons, embedding_dim)
        static_shape = list(input_shape)
        n_muons = static_shape[1]
        if n_muons is None:
            raise ValueError("The muon dimension (n_muons) must be fully defined.")
        # Create one learnable weight per muon.
        self.agg_weights = self.add_weight(
            shape=(n_muons, 1),
            initializer="uniform",
            trainable=True,
            name="aggregation_weights",
        )
        super().build(input_shape)

    def call(self, inputs):
        # Normalize the weights along the muon axis.
        weights = tf.nn.softmax(self.agg_weights, axis=0)
        # Multiply each muon's embedding by its weight and sum over muons.
        weighted_sum = tf.reduce_sum(
            inputs * weights, axis=1
        )  # (batch_size, embedding_dim)
        return weighted_sum


class AttentionAggregation(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.query = None

    def build(self, input_shape):
        # input_shape: (batch_size, n_muons, embedding_dim)
        static_shape = list(input_shape)
        embedding_dim = static_shape[-1]
        if embedding_dim is None:
            raise ValueError("The embedding dimension must be fully defined.")
        self.query = self.add_weight(
            shape=(embedding_dim,),
            initializer="uniform",
            trainable=True,
            name="query_vector",
        )
        super().build(input_shape)

    def call(self, inputs):
        # Compute attention scores as a dot product with the query.
        scores = tf.reduce_sum(inputs * self.query, axis=-1)  # (batch_size, n_muons)
        attention_weights = tf.nn.softmax(
            scores, axis=1
        )  # Normalize scores along the muon axis.
        # Compute weighted sum of muon embeddings.
        weighted_sum = tf.reduce_sum(
            inputs * tf.expand_dims(attention_weights, -1), axis=1
        )
        return weighted_sum  # (batch_size, embedding_dim)


class HybridAggregation(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create the submodules without fixed parameters.
        self.nn_agg = (
            NeuralNetworkAggregation()
        )  # You may choose a default output_dim if needed.
        self.attention_agg1 = AttentionAggregation()
        self.attention_agg2 = AttentionAggregation()

    def call(self, inputs):
        # Compute fixed aggregations along the muon axis.
        mean_agg = tf.reduce_mean(inputs, axis=1)
        max_agg = tf.reduce_max(inputs, axis=1)
        min_agg = tf.reduce_min(inputs, axis=1)
        # Compute a trainable aggregation using NeuralNetworkAggregation.
        nn_agg_out = self.nn_agg(inputs)
        # Concatenate all aggregated outputs.
        return tf.concat([mean_agg, max_agg, min_agg, nn_agg_out], axis=-1)


# class HybridAggregation(keras.layers.Layer):
#     def __init__(self, embedding_dim, **kwargs):
#         super(HybridAggregation, self).__init__(**kwargs)
#         self.trainable_agg = AttentionAggregation(embedding_dim=embedding_dim)
#         self.nn_agg = NeuralNetworkAggregation(output_dim=embedding_dim)
#         self.attention_agg = AttentionAggregation(embedding_dim=embedding_dim)

#     def call(self, inputs):
#         # Fixed aggregations
#         mean_agg = tf.reduce_mean(inputs, axis=1)
#         max_agg = tf.reduce_max(inputs, axis=1)
#         min_agg = tf.reduce_min(inputs, axis=1)
#         # Trainable aggregation
#         # trainable_agg = self.trainable_agg(inputs)
#         nn_agg = self.nn_agg(inputs)
#         # Concatenate all
#         return tf.concat([mean_agg, max_agg, min_agg, nn_agg], axis=-1)


class SimpleAggregation(keras.layers.Layer):
    # def __init__(self, embedding_dim, **kwargs):
    #     super().__init__(**kwargs)
    def call(self, inputs):
        # Fixed aggregations
        mean_agg = tf.reduce_mean(inputs, axis=1)
        max_agg = tf.reduce_max(inputs, axis=1)
        min_agg = tf.reduce_min(inputs, axis=1)
        return tf.concat([mean_agg, max_agg, min_agg], axis=-1)


class SumAggregation(keras.layers.Layer):
    # def __init__(self, embedding_dim, **kwargs):
    #     super().__init__(**kwargs)
    def call(self, inputs):
        # Fixed aggregations
        sum_agg = tf.reduce_sum(inputs, axis=1)
        # max_agg = tf.reduce_max(inputs, axis=1)
        # min_agg = tf.reduce_min(inputs, axis=1)
        return tf.concat([sum_agg], axis=-1)


class SimpleAggregation3(keras.layers.Layer):
    # def __init__(self, embedding_dim, **kwargs):
    #     super().__init__(**kwargs)
    def call(self, inputs):
        # Fixed aggregations
        mean_agg = tf.reduce_mean(inputs, axis=1)
        max_agg = tf.reduce_max(inputs, axis=1)
        min_agg = tf.reduce_min(inputs, axis=1)
        sum_agg = tf.reduce_sum(inputs, axis=1)
        std_agg = tf.math.reduce_std(inputs, axis=1)

        return tf.concat([mean_agg, max_agg, min_agg, sum_agg, std_agg], axis=-1)


class SimpleAggregation2(keras.layers.Layer):
    # def __init__(self, embedding_dim, **kwargs):
    #     super().__init__(**kwargs)
    # def call(self, inputs):
    #     # Fixed aggregations
    #     clean_inputs = tf.boolean_mask(inputs, tf.math.is_finite(inputs))
    #     mean_agg = tf.reduce_mean(clean_inputs, axis=1)
    #     max_agg = tf.reduce_max(clean_inputs, axis=1)
    #     min_agg = tf.reduce_min(clean_inputs, axis=1)
    #     return tf.concat([mean_agg, max_agg, min_agg], axis=-1)
    # def call(self, inputs):
    #     # Replace NaNs with 0 for mean computation
    #     is_finite = tf.math.is_finite(inputs) & (inputs != 0)
    #     masked_inputs = tf.where(is_finite, inputs, tf.zeros_like(inputs))

    #     # Avoid division by zero in mean calculation
    #     valid_counts = tf.reduce_sum(tf.cast(is_finite, tf.float32), axis=1, keepdims=True)
    #     mean_agg = tf.reduce_sum(masked_inputs, axis=1) / tf.maximum(valid_counts, 1.0)

    #     # Replace NaNs with -inf/inf for max/min calculations (so they are ignored)
    #     masked_max = tf.where(is_finite, inputs, tf.fill(tf.shape(inputs), -tf.float32.max))
    #     masked_min = tf.where(is_finite, inputs, tf.fill(tf.shape(inputs), tf.float32.max))

    #     max_agg = tf.reduce_max(masked_max, axis=1)
    #     min_agg = tf.reduce_min(masked_min, axis=1)

    #     return tf.concat([mean_agg, max_agg, min_agg], axis=-1)
    def call(self, inputs):
        # mean_agg = tf.reduce_mean(tf.where(tf.math.is_finite(inputs), inputs, 0), axis=1) #valid_counts
        is_valid = tf.math.is_finite(inputs) & (inputs != 0)
        valid_counts = tf.reduce_sum(tf.cast(is_valid, tf.float32), axis=1)

        masked_inputs = tf.where(is_valid, inputs, tf.zeros_like(inputs))
        mean_agg = tf.reduce_sum(masked_inputs, axis=1) / valid_counts
        # max_agg = tf.reduce_max(tf.where(tf.math.is_finite(inputs), inputs, -tf.float32.max), axis=1)
        # min_agg = tf.reduce_min(tf.where(tf.math.is_finite(inputs), inputs, tf.float32.max), axis=1)
        max_agg = tf.reduce_max(tf.where(is_valid, inputs, -tf.float32.max), axis=1)
        min_agg = tf.reduce_min(tf.where(is_valid, inputs, tf.float32.max), axis=1)
        return tf.concat([mean_agg, max_agg, min_agg], axis=-1)

    # def call(self, inputs):
    #     # Create a mask for valid (finite & non-zero) values
    #     is_valid = tf.math.is_finite(inputs) & (inputs != 0)

    #     # Compute valid count per row (for mean normalization)
    #     valid_counts = tf.reduce_sum(tf.cast(is_valid, tf.float32), axis=1, keepdims=True)

    #     # Replace NaNs and zeros with neutral values for aggregation
    #     masked_inputs = tf.where(is_valid, inputs, tf.zeros_like(inputs))

    #     # Mean aggregation (avoid division by zero)
    #     mean_agg = tf.reduce_sum(masked_inputs, axis=1) / tf.maximum(valid_counts, 1.0)

    #     # Max and Min ignoring NaNs (set invalid values to extreme numbers)
    #     max_agg = tf.reduce_max(tf.where(is_valid, inputs, -tf.float32.max), axis=1)
    #     min_agg = tf.reduce_min(tf.where(is_valid, inputs, tf.float32.max), axis=1)

    #     return tf.concat([mean_agg, max_agg, min_agg], axis=-1)

    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0], input_shape[2] * 3)


AGGREGATION_METHODS = {
    "simple": SimpleAggregation,
    "sum": SumAggregation,
    "simple3": SimpleAggregation3,
    "simple2": SimpleAggregation2,
    "weighted_sum": WeightedSumAggregation,
    "attention": AttentionAggregation,
    "neural_network": NeuralNetworkAggregation,
    "hybrid": HybridAggregation,
}

if __name__ == "__main__":
    DROPOUT = 0.3
    layers_list = [64]

    event_branch_kwargs = dict(
        layers_list=[256, 128] + layers_list,
        dropout=DROPOUT,
        kernel_regularizer=1e-3,
    )
    muon_branch_kwargs = dict(
        layers_list=layers_list,
        dropout=DROPOUT,
        kernel_regularizer=1e-3,
    )
    event_branch, muon_branch, mf.model = build_combined_model(
        event_feat_dim=n_event_features,
        muon_embedding_dim=32,
        layers_list=[64] + layers_list,
        event_branch_kwargs=event_branch_kwargs,
        muon_branch_kwargs=muon_branch_kwargs,
        dropout=DROPOUT,
    )

    # event_branch.summary()
    # muon_branch.summary()
    mf.xy_maker = xy_maker_muon_embedding
    mf.weights = "balanced_weights_CscdBDT"
    mf.optimizer = "adamw"
    mf.clipnorm = None
    mf.loss = "binary_crossentropy"

    mf.n_epoch = 30

    mf.compile_model()
    mf.fit()
