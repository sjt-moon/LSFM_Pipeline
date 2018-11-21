import re


def load_list(config):
    """
    Load list from configuration.

    Parameters:
        config (string): list of floats in string format

    Returns:
        list (list of floats)
    """
    return list(map(lambda x: float(x), filter(None, re.split(r"[^a-zA-Z0-9]+", config))))


def load_bool(config):
    """
    Load list from configuration.

    Parameters:
       config (string): boolean in string format

    Returns:
       True or False (boolean)
    """
    return config.lower() == 'true'


def load_number(config):
    """
    Load list from configuration.

    Parameters:
       config (string): number in string format

    Returns:
       Number value (float)
    """
    return float(config)
