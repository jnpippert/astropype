"""TODO docstring"""

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from scipy import ndimage

from .decorator import timeit
from .funcs import init_pool, remove_bad_pixel_func
from .pixelmath import clip_distribution, invert_mask, replace_nans


@timeit
def create_bad_pixel_mask(
    reference_file: Path, k: int = 5, output: str = "bad_pixel_mask.fits"
) -> np.ndarray:
    """TODO docstring"""
    print("detecting bad pixels...")
    ref_data: np.ndarray = fits.getdata(reference_file)
    clipped_ref_data = clip_distribution(ref_data.copy())
    filtered_data = ndimage.median_filter(ref_data, size=2)
    residual = ref_data - filtered_data
    clipped_residual = clip_distribution(
        residual.copy(), k=k, med=0, std=np.nanstd(clipped_ref_data)
    )
    clipped_residual *= 0
    bad_pixel_mask = invert_mask(replace_nans(clipped_residual, value=1))
    fits.writeto(output, bad_pixel_mask.astype(int), overwrite=True)
    return bad_pixel_mask


@timeit
def remove_bad_pixels(
    __files: list[Path],
    __mask_file: Path,
    remove: bool = True,
    prefix: str = "n",
    value: Any = None,
    value_func: str = None,
) -> list:
    """TODO docstring"""
    mask = fits.getdata(__mask_file).astype(np.float32)
    num = len(np.where(mask == 0)[0])
    mask[mask == 0] = np.nan
    print(f"removing {num} bad (Nan) pixels from file:")
    kwargs = {
        "prefix": prefix,
        "num": num,
        "mask": mask,
        "value": value,
        "value_func": value_func,
        "func": remove_bad_pixel_func,
        "remove": remove,
    }
    return init_pool(__files, kwargs)
