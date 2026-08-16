from astropy.io import fits
from numpy import ndarray
from .decorator import timeit
from scipy import ndimage
import numpy as np
from .pixelmath import invert_mask, replace_nans, fit_sky_model
from .pool import init_pool
from .funcs import (
    scale_func,
    starmask_func,
    starclip_func,
    dust_func,
    expand_func,
)
from pathlib import Path

__all__ = [
    "fit_sky_model",
    "create_masterflat_bad_pixel_mask",
    "scale_images",
    "create_flat_star_mask",
    "star_clipping",
    "remove_masked_dust",
    "expand_and_clean_mask",
]


@timeit
def create_masterflat_bad_pixel_mask(reference_data: ndarray, k: int = 3) -> ndarray:
    print("detecting bad pixels...")
    filtered_data = ndimage.median_filter(reference_data, size=2)
    residual = reference_data - filtered_data
    std = np.std(residual)
    residual[residual > k * std] = np.nan
    residual *= 0
    bad_pixel_mask = invert_mask(replace_nans(residual, value=1)).astype(np.float32)
    return bad_pixel_mask


@timeit
def scale_images(__files: list, __reference_file: Path, remove : bool = False, prefix: str = "scaled_"):
    print(f"scale by {__reference_file} for")
    ref_data = fits.getdata(__reference_file)
    kwargs = {
        "prefix": prefix,
        "reference_file": __reference_file,
        "reference_data": ref_data,
        "func": scale_func,
        "remove" : remove
    }
    return init_pool(__files, kwargs)


@timeit
def create_flat_star_mask(__files: list, k: int = 3, remove : bool = False, prefix="starmask_"):
    print(f"create star mask using background fitting for")
    kwargs = {
        "prefix": prefix,
        "k": k,
        "func": starmask_func,
        "remove" : remove,
    }
    return init_pool(__files, kwargs)


@timeit
def star_clipping(__files: list, __maskfiles: list, remove : bool = False, prefix: str = "k"):
    print(f"clipping stars based on extended star mask for files:")
    kwargs = {
        "prefix": prefix,
        "maskfiles": __maskfiles,
        "func": starclip_func,
        "remove" : remove,
    }
    return init_pool(__files, kwargs)


@timeit
def remove_masked_dust(__maskfiles, remove : bool = False, prefix: str = "r"):
    print("removing masked dust from files:")
    masks = [fits.getdata(file) for file in __maskfiles]
    dust_mask = invert_mask(np.sum(masks, axis=0)).astype(int)
    kwargs = {
        "prefix": prefix,
        "maskfiles": __maskfiles,
        "dustmask": dust_mask,
        "func": dust_func,
        "remove" : remove,
    }
    return init_pool(__maskfiles, kwargs)


@timeit
def expand_and_clean_mask(__maskfiles, diameter: int = 3,remove : bool = False, prefix: str = "e"):
    print(f"expanding star mask with diameter of {diameter} ...")
    kwargs = {
        "prefix": prefix,
        "diameter": diameter,
        "func": expand_func,
        "remove" : remove,
    }
    return init_pool(__maskfiles, kwargs)
