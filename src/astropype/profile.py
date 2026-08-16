from __future__ import annotations

from pathlib import Path
import yaml

__all__ = ["load_profile"]

_NULL_TOKENS = {"none", "null", "~"}


def _normalize(value):
    if isinstance(value, dict):
        return {key: _normalize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize(val) for val in value]
    if isinstance(value, str) and value.lower() in _NULL_TOKENS:
        return None
    return value


def load_profile(path: Path | str) -> dict:
    """
    Loads a camera or telescope profile YAML file.

    Treats the strings 'None'/'null'/'~' (any case) the same as YAML's
    real null, since a profile hand-edited in a text editor may use
    either spelling.

    Parameters
    ----------
    path : Path, str
        Path to the profile YAML file.

    Returns
    -------
    profile : dict
        The parsed profile, with null-like strings normalized to None.
    """
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return _normalize(data)
