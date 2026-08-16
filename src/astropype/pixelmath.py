import numpy as np
from typing import Any, Literal


__all__ = ["invert_mask", "circle_mask", "clip_distribution", "replace_nans"]


def invert_mask(mask: np.ndarray):
    """
    Inverts a mask. Non-zero values will be inverted to zero.

    Parameters
    ----------
    mask : np.ndarray
        The mask array to invert.

    Returns
    -------
    inverted mask : np.ndarray
        The inverted mask array.

    Raises
    ------
    TypeError
        If ``mask`` is not of type numpy.ndarray.
    """
    if not isinstance(mask, np.ndarray):
        TypeError(f"'mask' of non-numpy.ndarray type {type(mask)}.")
    return ~mask.astype(bool)


def circle_mask(radius: int) -> np.ndarray:
    """
    Creates a circular kernel mask of a given radius.

    Parameters
    ----------
    radius : int
        The radius of the circle.

    Returns
    -------
    circle mask : np.ndarray
        An array with odd shape containing a centered circle
        with values of 1.

    Raises
    ------
    TypeError
        If ``radius`` is not of type integer.
    """
    if not isinstance(radius, int):
        raise TypeError(f"'radius' of non-int type {type(radius)}.")
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    bool_mask = np.sqrt((x) ** 2 + (y) ** 2) <= radius
    return bool_mask.astype(int)


def clip_distribution(
    distribution: np.ndarray,
    k: int = 3,
    med: float = None,
    std: float = None,
    nan_policy: Literal["ignore", "remove"] = "ignore",
) -> np.ndarray:
    """
    Kappa-Sigma clips a given distribution of values.

    Parameters
    ----------
    distribution : np.ndarray
        The value distribution.

    k : int, optional
        The kappa value of the clipping. Default is 3.

    nan_policy : str, optional
        Used to handle the clipped (nan) values.
        'ignore' leaves the nan values in the distribution.
        'remove' removes the nan values from the distribution.
        Default is 'ignore'

    Returns
    -------
    clipped distribution : np.ndarray
        The clipped distribution, where clipped pixels are numpy.nan values.

    Raises
    ------
    TypeError
        If ``distribution`` is not of type numpy.ndarray.

    ValueError
        If a false value for ``nan_policy`` is given.
    """
    if not isinstance(distribution, np.ndarray):
        TypeError(f"'distribution' of non-numpy.ndarray type {type(distribution)}.")
    if not isinstance(med, float):
        med = np.median(distribution)
    if not isinstance(std, float):
        std = np.std(distribution, ddof=1)
    distribution[distribution > med + k * std] = np.nan
    distribution[distribution < med - k * std] = np.nan
    if nan_policy == "ignore":
        return distribution
    if nan_policy == "remove":
        return distribution[~np.isnan(distribution)]
    raise ValueError("invalid value for 'nan_policy'.")


def replace_nans(
    array: np.ndarray, value: Any = None, value_func: str = None, radius: int = 1
):
    """
    Replaces nan values in an array.

    Parameters
    ----------
    array : np.ndarray
        The array where to replace nan values.

    value : Any, int, float, optional
        Constant value to replace nan values with.
        If not None, this method activates first directly returns
        afterwards. Default is None.

    value_func : str, optional
        The function to determine the replace value.
        'median' uses the median of the whole array.
        'mean' uses the mean of the whole array.
        'normal' uses random drawn values from the statistic of
        the array.
        'nearest' takes the median of the sorrounding pixels with
        a given radius. Default is None.

    radius : int, optional
        The size of the square to compute the median of the nearest neighbours.
        Default is 1.
    """
    if value is not None:
        return np.nan_to_num(array, nan=value)
    if value_func == "median":
        return np.nan_to_num(array, nan=np.nanmedian(array))
    if value_func == "mean":
        return np.nan_to_num(array, nan=np.nanmean(array))
    nan_ids = np.argwhere(np.isnan(array) == True)
    if value_func == "normal":
        std = np.nanstd(array)
        avg = np.nanmean(array)
        for px in nan_ids:
            array[px] = np.random.normal(avg, std)
        return array
    if value_func == "nearest":
        array = np.pad(array, radius, mode="empty")
        for y, x in nan_ids:
            y += radius
            x += radius
            array[y][x] = np.nanmedian(
                array[y - radius : y + radius + 1, x - radius : x + radius + 1]
            )
        return array[radius:-radius, radius:-radius]
    print("[WARNING] no replacing method applied. Returning unchanged array.")
    return array
