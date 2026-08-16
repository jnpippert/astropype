import os
from pathlib import Path
from typing import Any
from astropy.io import fits
from tqdm import tqdm
import pandas as pd
__all__ = ["contains_substring", "get_filepaths", "copy_files", "rename_files", "remove_files"]


def contains_substring(string: str, contents: list) -> bool:
    """
    Checks if a string contains certain sub-strings in true
    order.

    Parameters
    ----------
    string : str
        The string to check.

    contents : list
        The list of sub-strings to check.

    Returns
    -------
    awnser : bool
        True if ``string`` contains all ``contents``. Otherwise False.

    Raises
    ------
    TypeError
        If contents is not of type list or str.
    """
    if isinstance(contents, str):
        contents = [contents]
    if not isinstance(contents, list):
        TypeError(f"'contents' of non-list type {type(contents)}.")
    pos = 0
    for content in contents:
        if (index := string.find(content)) < pos:
            return False
        pos = index
    return True


def get_filepaths(path: Path, identifier: str = "*") -> list:
    """
    Determines all filepaths in a given directory regarding certain
    identifiers.

    Returning files with given prefix 'x'.
    >>> get_filepaths(".", identifier = "x*")

    Returning files with given suffix 'x'.
    >>> get_filepaths(".", identifier = "*x")


    Returning files with given sub-string 'x'.
    >>> get_filepaths(".", identifier = "*x*")

    Returning files with given sub-strings 'x','y' and 'z'.
    >>> get_filepaths(".", identifier = "*x*y*z*")

    The order of the sub-strings is maintained.

    Parameters
    ----------
    path : Path, str
        The path of the directory to check.

    identifier : str
        The sub-strings seperated by asterisk. Default is '*' which
        results in returning all files in the directory.

    Returns
    -------
    files : list
        List of paths to the selected files.

    Raises
    ------
    TypeError
        If path is not of type Path or str.
    """
    if isinstance(path, str):
        path = Path(str)
    if not isinstance(path, Path):
        TypeError(f"'path' of non-Path type {type(path)}.")
    sub_strings = identifier.split("*")
    prefix = sub_strings[0]
    suffix = sub_strings[-1]
    files = []
    for file in path.iterdir():
        filename = file.name
        if (
            filename.startswith(prefix)
            and filename.endswith(suffix)
            and contains_substring(filename, sub_strings[1:-1])
        ):
            files.append(file)
    return files


def copy_files(files: list, destination: Path) -> None:
    """
    Copies files to a given destination.

    Parameters
    ----------
    files : list,str
        List or filename of files to copy.

    destination : Path
        Path to the destination directory.

    Raises
    ------
    TypeError
        If files is not of type list or str.
    """
    if not isinstance(files, list):
        files = [files]
    if not isinstance(files, list):
        TypeError(f"'files' of non-list type {type(files)}.")
    print(f"[INFO] Copying files to {destination} ...")
    for file in (bar:=tqdm(files)):
        bar.set_description(f"{file}")
        bar.refresh()
        os.system(f"cp {Path(file)} {destination}")

def remove_files(files: list) -> None:
    """
    Remove files.

    Parameters
    ----------
    files : list,str
        List or filename of files to copy.

    Raises
    ------
    TypeError
        If files is not of type list or str.
    """
    if not isinstance(files, list):
        files = [files]
    if not isinstance(files, list):
        TypeError(f"'files' of non-list type {type(files)}.")
    print(f"[INFO] Removing files ...")
    for file in (bar:=tqdm(files)):
        bar.set_description(f"{file}")
        bar.refresh()
        os.system(f"rm -f {Path(file)}")

def rename_files(path: Path, __date: str, __flatdate: str):
    files = [f for f in path.iterdir() if f.is_file()]
    files = sort_files_by_obsdate(files)
    for i, file in (bar:=tqdm(enumerate(files),total=len(files))):
        __object = fits.getval(file, "OBJECT")
        __filter = fits.getval(file, "FILTER")
        __count = f"{(3-len(str(i+1)))*'0'}{i+1}"
        if __object == "bias_pipeline":
            __object = "bias"
        if __object == "dark_pipeline":
            __object = "dark"
        if __object == "sky":
            new_filename = (
                f"{__object.lower()}_{__filter}_qhy_{__flatdate}_{__count}.fits"
            )
        else:
            new_filename = (
                f"{__object.lower()}_{__filter}_qhy_{__date}_{__count}.fits"
            )
        bar.set_description(f"{file.name} -> {new_filename}")
        bar.refresh()
        os.system(f"mv {file} {file.parent.joinpath(new_filename)}")

def sort_files_by_obsdate(files : list[Path]) -> list[Path]:
    scheme = {"path" : "str",
              "filename" : "str",
              "date" : "datetime64[ns]"}
    df = pd.DataFrame(columns=scheme.keys()).astype(scheme)
    for f in files:
        df.loc[len(df.index)] = [f.parent, f.name, pd.Timestamp(fits.getval(f,"DATE-OBS"))]
    print(df)
    df = df.sort_values(by=["date"],ascending=True)
    return [Path(row["path"]).joinpath(row["filename"]) for index,row in df.iterrows()]
