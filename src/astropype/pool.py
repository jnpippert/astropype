import os
from pathlib import Path
from multiprocessing import Pool
from astropy.io import fits

__all__ = ["init_pool"]


def init_pool(files: list, kwargs: dict) -> list:
    if isinstance(files, (Path, str)):
        files = [files]
    with Pool() as pool:
        result = pool.map(
            pool_func,
            [
                (
                    file,
                    kwargs,
                )
                for file in files
            ],
        )
    return result


def pool_func(args: list):
    """
    The general method to apply multiprocessing in the data reduction.
    Used as the 'func' parameter in multiprocessing.Pool.map().

    Generally ``args`` holds the pooling function and keyword arguments.
    The ``kwargs`` always contain the refernence to the reduction function
    and the prefix of the new filename. All other ``kwargs`` are specific to
    the reduction step and reduction function.

    >>> with Pool() as pool:
    >>>     result = pool.map(pool_func, {'file' : filepath,
    >>>                                   'prefix' : prefix ,
    >>>                                   'other' : value})

    Parameters
    ----------
    args : list
        A list of arguments. First item holds the file path.
        The second item is a dictionary of keyword arguments which are used to carry
        further parameters such as the new filename prefix, the reduction func.

    Returns
    -------
    filename : Path
        Path of the new reduced/modified file.
    """
    file, kwargs = args
    if isinstance(file, str):
        file = Path(file)
    print(f"\t{file}")
    new_filename = file.with_name(kwargs["prefix"] + file.name)
    data, header = kwargs["func"](file, kwargs)
    fits.writeto(new_filename, data, header, overwrite=True)

    try:
        os.remove(file) if kwargs["remove"] else None
        print(f"\t\t... {file.name} removed")
    except FileNotFoundError:
        print(f"[INFO] No file or directory: ... {file.name}")

    return new_filename
