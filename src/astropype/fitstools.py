from pathlib import Path
from .decorator import timeit
from .logger import logger
from .pool import init_pool
from .funcs import (
    subtract_func,
    divide_func,
    rotate_func,
    crop_func,
    overscan_func,
    bin_func,
)


@timeit
def subtractfits(__files: list, __file: Path, remove : bool = False, prefix: str = "s") -> list:
    logger.info(f"subtracting {__file} from:")
    kwargs = {"reference_file": __file, "prefix": prefix, "func": subtract_func, "remove" : remove}
    return init_pool(__files, kwargs)


@timeit
def dividefits(__files: list, __file: Path, remove : bool = False, prefix: str = "d") -> list:
    logger.info(f"dividing {__file} from:")
    kwargs = {"reference_file": __file, "prefix": prefix, "func": divide_func, "remove" : remove}
    return init_pool(__files, kwargs)


@timeit
def rotatefits(__files: list, remove : bool = False, prefix: str = "r") -> list:
    logger.info(f"rotating frames ...")
    kwargs = {"prefix": prefix, "func": rotate_func, "remove" : remove}
    return init_pool(__files, kwargs)


@timeit
def cropfits(__files: list, crop_rows=None, crop_cols=None, remove : bool = False, prefix: str = "c"):
    if crop_rows is None or crop_cols is None:
        logger.info("No crop region defined - skipping crop step.")
        return __files
    logger.info(f"cropping overscan region ...")
    kwargs = {"prefix": prefix, "func": crop_func, "remove" : remove,
              "crop_rows": crop_rows, "crop_cols": crop_cols}
    return init_pool(__files, kwargs)


@timeit
def subtract_overscan(__files: list, overscan_rows=None, overscan_cols=None, remove : bool = False, prefix: str = "o"):
    if overscan_rows is None or overscan_cols is None:
        logger.info("No overscan region defined - skipping overscan subtraction.")
        return __files
    logger.info(f"subtracting individual overscans ...")
    kwargs = {"prefix": prefix, "func": overscan_func, "remove" : remove,
              "overscan_rows": overscan_rows, "overscan_cols": overscan_cols}
    return init_pool(__files, kwargs)


@timeit
def binfits(__files : list, bin_factor : int, bin_method : str = "sum",
            consider_nans : bool = False, remove : bool = False, prefix : str = "b"):
    logger.info(f"binning images by a factor of {bin_factor} ...")
    kwargs = {"prefix" : f"{prefix}{bin_factor}", "func" : bin_func, "bin_factor" : bin_factor,
              "bin_method" : bin_method , "consider_nans" : consider_nans, "remove" : remove}
    return init_pool(__files,kwargs)
