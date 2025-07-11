import os
from pathlib import Path
import yaml
import getpass
# Path to the config file in the user's home directory
CONFIG_FILE_PATH = Path.home() / ".i3kiss.config.yaml"

username = getpass.getuser()

DEFAULT_DIRECTORIES = {
        "model_base_dir": f"/data/user/{username}/i3kiss/models/",
        "config_base_dir": f"/data/user/{username}/i3kiss/configs/",
        "plot_base_dir": f"/data/user/{username}/plots/",
        "fname_train" : "train.hdf5",
        "fname_test" : "test.hdf5",
        "data_base_dir": f"/data/user/{username}/", 
}

def create_default_config():
    """Create a configuration file with user-provided or default values."""
    print("Creating a new configuration file...")
    #config = {'directories': {}}
    config = {}
    
    # Prompt the user for each path
    for key, default_value in DEFAULT_DIRECTORIES.items():
        user_input = input(f"Enter the path for {key.replace('_', ' ')} (default: {default_value}): ").strip()
        config[key] = user_input if user_input else default_value
    
    # Write the config to the file
    # with open(CONFIG_FILE_PATH, "w") as config_file:
    #     config.write(config_file)
    with open(CONFIG_FILE_PATH, "w") as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

    print(f"Configuration file created at {CONFIG_FILE_PATH}.")
    print("You can edit this file later if needed.")


def load_user_config():
    """Load the default configuration file, or create one if it doesn't exist."""    
    if not CONFIG_FILE_PATH.exists():
        print(f"Configuration file not found at {CONFIG_FILE_PATH}.")
        create = input("Do you want to create a configuration file? [y/n]: ").strip().lower()
        if create == "y":
            create_default_config()
        else:
            raise FileNotFoundError("Configuration file is required. Exiting.")
    
    with open(CONFIG_FILE_PATH, "r") as config_file:
        return yaml.safe_load(config_file)


def load_config(config_file_path):
    """Load the configuration from the file."""
    
    with open(config_file_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    default_config = load_user_config()

    # Update the default config with the user-provided values
    for key, value in config.items():
        default_config[key] = value

    for key, value in default_config.items():
        if isinstance(value, str) and '~' in value:
            default_config[key] = os.path.expanduser(value)

    return default_config

script_dir = os.path.dirname(__file__)
default_config_path = os.path.join(script_dir, "configs", "defaults.yml")
default_config = load_config(default_config_path)

# if __name__ == "__main__":
#     config = load_default_config()


