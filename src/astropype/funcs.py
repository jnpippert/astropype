from astropy.io import fits
from pathlib import Path
from multiprocessing import Pool
from typing import Any
from astropy.io.fits import Header
from numpy import ndarray
import numpy as np
import os
from astropype.pixelmath import (
    replace_nans,
    clip_distribution,
    invert_mask,
    circle_mask,
)

__all__ = [
    "init_pool",
    "subtract_func",
    "divide_func",
    "rotate_func",
    "overscan_func",
    "scale_func",
    "starmask_func",
    "starclip_func",
    "dust_func",
    "expand_func",
    "bin_func",
]


def init_pool(files: list, kwargs: dict) -> list:
    if isinstance(files, (Path, str)):
        files = [files]
    with Pool(processes=2) as pool:
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
    new_filename = f"{str(file).replace(file.name,kwargs['prefix']+file.name)}"
    data, header = kwargs["func"](file, kwargs)
    fits.writeto(new_filename, data, header, overwrite=True)
    
    try:
        os.remove(file) if kwargs["remove"] else None
        print(f"\t\t... {file.name} removed")
    except FileNotFoundError:
        print(f"[INFO] No file or directory: ... {file.name}")
    
    return new_filename


def subtract_func(file: Path, kwargs: dict) -> tuple:
    data, header = fits.getdata(file, header=True)
    header["HISTORY"] = "----------"
    header["HISTORY"] = f"subtracted {kwargs['reference_file']}"
    header["HISTORY"] = f"added file prefix: {kwargs['prefix']}"
    data -= fits.getdata(kwargs["reference_file"])
    return data, header


def divide_func(file: Path, kwargs: dict) -> tuple:
    data, header = fits.getdata(file, header=True)
    header["HISTORY"] = "----------"
    header["HISTORY"] = f"divided {kwargs['reference_file']}"
    header["HISTORY"] = f"added file prefix: {kwargs['prefix']}"
    data /= fits.getdata(kwargs["reference_file"])
    return data, header


def rotate_func(file: Path, kwargs: dict) -> tuple:
    data, header = fits.getdata(file, header=True)
    if header["TRACK"] in ["1"]:
        data = np.rot90(data, 2)
        header["CD_1_1"] = -header["CD1_1"]
        header["CD_1_2"] = -header["CD1_2"]
        header["CD_2_1"] = -header["CD2_1"]
        header["CD_2_2"] = -header["CD2_2"]
        header["HISTORY"] = f"rotated image by 180 degree"
    else:
        header["HISTORY"] = "rotated image by 0 degree"
    header["HISTORY"] = f"added file prefix: {kwargs['prefix']}"
    return data, header


def crop_func(file: Path, kwargs: dict) -> tuple:
    data, header = fits.getdata(file, header=True)
    data = data[0:6388, 24:]
    header["HISTORY"] = f"cropped image [0:6388, 24:]"
    header["HISTORY"] = "new size: (6388,9576)"
    header["HISTORY"] = f"added file prefix: {kwargs['prefix']}"
    return data, header


def overscan_func(file: Path, kwargs: dict) -> tuple:
    data, header = fits.getdata(file, header=True)
    data = data.astype(np.float32)
    overscan = np.median(data[0:6388, 0:10])
    data -= overscan
    header["HISTORY"] = f"subtracted overscan offset"
    header["HISTORY"] = f"offset value: {overscan}"
    header["HISTORY"] = f"added file prefix: {kwargs['prefix']}"
    return data, header


def remove_bad_pixel_func(file: Path, kwargs: dict) -> tuple:
    data, header = fits.getdata(file, header=True)
    header["HISTORY"] = f"replaced {kwargs['num']} bad (Nan) pixels"
    data *= kwargs["mask"]
    data = replace_nans(data, value=kwargs["value"], value_func=kwargs["value_func"])
    header["HISTORY"] = f"added file prefix: {kwargs['prefix']}"
    return data, header


def scale_func(file: Path, kwargs: dict) -> tuple:
    data, header = fits.getdata(file, header=True)
    scale = np.nanmedian(kwargs["reference_data"] / data)
    header["HISTORY"] = f"scale reference image: {kwargs['reference_file']}"
    header["HISTORY"] = f"scale factor: {scale}"
    header["HISTORY"] = f"added file prefix: {kwargs['prefix']}"
    data *= scale
    return data, header


def starmask_func(file: Path, kwargs: dict) -> tuple:
    from astropype.sky import fit_sky_model

    data, header = fits.getdata(file, header=True)
    sky = fit_sky_model(data)
    residual = data - sky
    std = np.nanstd(clip_distribution(residual, k=kwargs["k"]))
    residual[residual > kwargs["k"] * std] = np.nan
    mask = invert_mask(replace_nans(residual * 0, value=1)).astype(int)
    data = mask
    header["HISTORY"] = "clipped values via background fitting"
    header["HISTORY"] = "transformed into mask"
    header["HISTORY"] = f"added prefix: {kwargs['prefix']}"
    return data, header


def starclip_func(file: Path, kwargs: dict) -> tuple:
    data, header = fits.getdata(file, header=True)
    for i, mask in enumerate(kwargs["maskfiles"]):
        if file.name in str(mask):
            break
    mask = fits.getdata(kwargs["maskfiles"][i])
    data *= mask
    data[data == 0] = np.nan
    header["HISTORY"] = f"clipped values via star mask: {kwargs['maskfiles'][i]}"
    header["HISTORY"] = f"added prefix: {kwargs['prefix']}"
    return data, header


def dust_func(file: Path, kwargs: dict) -> tuple:
    data, header = fits.getdata(file, header=True)
    data += kwargs["dustmask"]
    header["HISTORY"] = "removed masked dust pixels from star mask"
    header["HISTORY"] = f"added prefix: {kwargs['prefix']}"
    return data, header


def expand_func(file: Path, kwargs: dict) -> tuple:
    data, header = fits.getdata(file, header=True)
    data = np.pad(data, kwargs["diameter"], mode="constant", constant_values=1)
    new_data = np.zeros(data.shape)
    zero_pixels = np.argwhere(data == 0)
    for xy in zero_pixels:
        y, x = xy
        if (
            np.sum(
                data[
                    y - 1 : y + 2,
                    x - 1 : x + 2,
                ]
            )
            >= 6
        ):
            continue
        expand_mask = circle_mask(kwargs["diameter"] // 2)
        new_data[
            y - kwargs["diameter"] // 2 : y + kwargs["diameter"] // 2 + 1,
            x - kwargs["diameter"] // 2 : x + kwargs["diameter"] // 2 + 1,
        ] += expand_mask
    new_data[new_data > 0] = 1
    data = invert_mask(
        new_data[
            kwargs["diameter"] : -kwargs["diameter"],
            kwargs["diameter"] : -kwargs["diameter"],
        ]
    ).astype(int)

    return data, header


def bin_func(file : Path, kwargs : dict) -> tuple:
    data,header = fits.getdata(file,header=True)
    m,n = data.shape
    mcut = -(data.shape[0]%kwargs["bin_factor"]) if data.shape[0]%kwargs["bin_factor"] > 0 else m
    ncut = -(data.shape[1]%kwargs["bin_factor"]) if data.shape[1]%kwargs["bin_factor"] > 0 else n
    bin_funcs = {"mean" : np.mean,"median" : np.median,"sum" : np.sum}
    if kwargs["consider_nans"]:
        bin_funcs = {"mean" : np.nanmean,"median" : np.nanmedian,"sum" : np.nansum}
    data = data[0:mcut,0:ncut]
    reshaped_data = data.reshape(m // kwargs["bin_factor"], kwargs["bin_factor"], 
                                 n // kwargs["bin_factor"], kwargs["bin_factor"])
    # wcs correction
    header["CD1_1"] *= kwargs["bin_factor"]
    header["CD1_2"] *= kwargs["bin_factor"]
    header["CD2_1"] *= kwargs["bin_factor"]
    header["CD2_2"] *= kwargs["bin_factor"]
    header["CRPIX1"] //= kwargs["bin_factor"]
    header["CRPIX2"] //= kwargs["bin_factor"]

    return bin_funcs[kwargs["bin_method"]](bin_funcs[kwargs["bin_method"]](reshaped_data,axis=1),axis=2), header
