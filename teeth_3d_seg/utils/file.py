import json


def load_json(file_path: str) -> dict:
    """Loading a json file

    Parameters
    ----------
    file_path : str
        path to the json file

    Returns
    -------
    dict
        json file
    """
    with open(file_path, "r") as f:
        json_file = json.load(f)

    return json_file
