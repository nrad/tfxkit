import utils

feature_sets = {
    "primary": [
        "log_z",
        "log_rho",
        "log_length",
        "log_energy",
        "interaction_height",
        "cos_zenith",
        "pdg_map",
        "log_energy_per_nucleon",
    ],
    "muons": [
        "depth",
        # "cos_theta",
        "multiplicity",
    ],
    "muons_radius_sum": [
        "mu_radius_mean",
        "mu_radius_std",
        "mu_radius_min",
        "mu_radius_max",
    ],
    "muons_log_energy_sum": [
        "mu_log_energy_mean",
        "mu_log_energy_std",
        "mu_log_energy_min",
        "mu_log_energy_max",
    ],
    "muons_log_energy": ["mu%s_%s" % (imu, "log_energy") for imu in range(1, 11)],
    "muons_radius": ["mu%s_%s" % (imu, "radius") for imu in range(1, 11)],
    "muon_bundle": [
        "singleness",
        "mu_leading_energy_fraction",
        "mu_bundle_log_energy",
    ],
}


DEFAULT_FEATURES = [
    "log_rho",
    "log_z",
    "log_length",
    "log_energy",
    "interaction_height",
    "cos_zenith",
    "pdg_map",
    "depth",
    # "cos_theta",
    "multiplicity",
    "mu_log_energy_max",
    "mu_log_energy_mean",
    "mu_log_energy_std",
    "mu_radius_mean",
    #
    "singleness",
    "mu_leading_energy_fraction",
    "mu_bundle_log_energy",
    "log_energy_per_nucleon",
]
mu_vars_sum = feature_sets["muons_log_energy_sum"] + feature_sets["muons_radius_sum"]

mu_vars = [
    "mu%s_%s" % (imu, v) for imu in range(1, 11) for v in ["log_energy", "radius"]
]

# DEFAULT_FEATURES += mu_vars_sum
# DEFAULT_FEATURES += mu_vars

DEFAULT_FEATURE_SETS = [
    "primary",
    "muons",
    "muons_radius_sum",
    "muons_log_energy_sum",
    "muons_radius",
    "muons_log_energy",
    "muon_bundle",
]


def select_features(features=DEFAULT_FEATURES, add_features=[], remove_features=[]):
    features_input = features + add_features
    features = []
    for feature in features_input:
        features.extend(feature_sets.get(feature, [feature]))

    features = utils.unique(features)
    for feature in remove_features:
        remove_list = feature_sets.get(feature, [feature])
        for to_remove in remove_list:
            if not to_remove in features:
                print(
                    "WARNING: Asked to remove a feature which was not included anyways %s"
                    % to_remove
                )
                continue
            features.pop(features.index(to_remove))
    return features
