import json


def read_params():
    """
    Reads a JSON-formatted file named "params.json" in the current working directory.

    Returns:
        A dictionary containing the contents of the JSON file.

    Raises:
        FileNotFoundError: If the file is not found.
        ValueError: If the file is not well-formed JSON.
    """
    try:
        with open("params.json", "r") as f:
            json_str = f.read()
        try:
            params = json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Your params.json file is badly formed")
            exit(1)
        return params
    except FileNotFoundError:
        print("You are missing the params.json file in your directory")
        exit(1)
