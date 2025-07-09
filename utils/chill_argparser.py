import argparse

def convert_to_int_or_float(value):
    """
    Convert a string to an integer or float if possible.
    Args:
        value (str): The string to convert.
    Returns:
        int or float or str: The converted value.
    """

    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

class ChillArgumentParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Base Argument Parser")
        self._recognized_args = set()  # To track recognized arguments

    def add_argument(self, *args, **kwargs):
        """
        Wrapper for adding recognized arguments.
        """
        # Extract argument names (e.g., --arg)
        self._recognized_args.update(arg.lstrip("-") for arg in args if arg.startswith("--"))
        self.parser.add_argument(*args, **kwargs)

    def parse_arguments(self):
        """
        Parse arguments and separate recognized and unknown arguments.
        Returns:
            known_args (dict): Dictionary of recognized arguments.
            unknown_args (dict): Dictionary of unrecognized arguments, with values converted to float if numeric.
        """
        known_args, unknown = self.parser.parse_known_args()
        
        # Convert known arguments to a dictionary
        # known_args = vars(args)

        # Parse unknown arguments into a dictionary
        unknown_args = {}
        key = None
        for item in unknown:
            if item.startswith("--"):  # Detect argument key
                key = item.lstrip("-")
                unknown_args[key] = []
            elif key:  # Collect values for the last key
                # Convert to float if numeric
                # value = float(item) if item.replace('.', '', 1).isdigit() else item
                print("------------- unknown_args")
                value = convert_to_int_or_float(item)
                print(key, value, type(value), len(str(value)) ) 
                unknown_args[key].append(value)
            else:
                raise ValueError(f"Unexpected argument format: {item}")

        # Convert single-item lists to scalars
        for k, v in unknown_args.items():
            if len(v) == 1:
                unknown_args[k] = v[0]

        return known_args, unknown_args

# Example usage
if __name__ == "__main__":
    class MyArgumentParser(ChillArgumentParser):
        def __init__(self):
            super().__init__()
            # Add additional recognized arguments here
            self.add_argument("--additional_arg", type=int, help="An additional recognized argument.")
    
    # Instantiate and parse arguments
    parser = MyArgumentParser()
    known_args, unknown_args = parser.parse_arguments()
    
    print("Known Arguments:", known_args)
    print("Unknown Arguments:", unknown_args)