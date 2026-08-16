from astropy.io import fits
import numpy as np
from .logger import logger
from .sky import create_masterflat_bad_pixel_mask
from .pixelmath import replace_nans
from photutils.background import Background2D, MedianBackground
from astropy.stats import SigmaClip
from pathlib import Path
from scipy import ndimage

MASTERHEADER = fits.Header()
MASTERHEADER["EXPTIME"] = (0, "exposure time in seconds")
MASTERHEADER["FILTER"] = ("D", "observed filter band")
MASTERHEADER["OBJECT"] = ("masterframe", "type of image")


__all__ = ["MasterBias", "MasterDark", "MasterFlat"]


class Master:
    def __init__(
        self, files: list, method: str = "mean", k: int = None, q: int = None
    ):
        self._files = files
        self.data = None
        self._header = MASTERHEADER.copy()
        self._method = method
        self._k = k
        self._q = q
        self._type = None
        self.path: Path = None
        if len(files) == 0:
            raise ValueError("empty list of files!")

    def _create(self):
        data = []
        if isinstance(self._k, (int, float)):
            raise NotImplementedError
        for file in self._files:
            data.append(fits.getdata(file))
        logger.info(
            f"computing master{self._type} using {self._method} combination method..."
        )
        if self._method == "mean":
            self.data = np.nanmean(data, axis=0, dtype=np.float32)
        if self._method == "median":
            self.data = np.nanmedian(data, axis=0)

    def writeto(self, filename: str, overwrite: bool = True):
        self.path = filename
        self._header["OBJECT"] = f"master{self._type}"
        if self.data is None:
            raise TypeError(f"master data of non-array type {type(self.data)}")
        logger.info(f"writing master{self._type} to file: '{filename}' ...")
        fits.writeto(
            filename=filename, data=self.data, header=self._header, overwrite=overwrite
        )


class MasterBias(Master):
    def __init__(
        self, files: list, method: str = "mean", k: int = None, q: int = None
    ):
        super().__init__(files=files, method=method, k=k, q=q)
        self._header["EXPTIME"] = 0
        self._header["FILTER"] = "D"
        self._type = "bias"
        self._create()


class MasterDark(Master):
    def __init__(
        self,
        files: list,
        method: str = "mean",
        exptime: float = None,
        temperature: float = None,
        k: int = None,
        q: int = None,
    ):
        super().__init__(files=files, method=method, k=k, q=q)
        self.data = np.zeros(fits.getdata(files[0]).shape)
        self._dont_create = True
        if isinstance(exptime, (int, float)):
            self._header["EXPTIME"] = exptime
            self.exptime = exptime
            self._check_exptime()
        if isinstance(temperature, (int, float)):
            self._header["TEMPERATUR"] = temperature
            self.temperature = temperature
            # self._check_temperature()
        self._header = MASTERHEADER.copy()
        self._header["FILTER"] = "D"
        self._type = "dark"
        if self._dont_create:
            self._create()

    def _check_exptime(self):
        i = 0
        while i < len(self._files):
            if (val := fits.getval(file := self._files[i], "EXPTIME")) != self.exptime:
                logger.info(f"{file} removed with wrong exposure time of {val}s")
                self._files.remove(file)
                continue
            i += 1
        if len(self._files) == 0:
            logger.warning("No dark frame with matchng exposure time!")
            logger.warning("Subtracting zero.")
            self._dont_create = False


class MasterFlat(Master):
    def __init__(
        self,
        files: list,
        method: str = "mean",
        k: int = None,
        q: int = None,
    ):
        super().__init__(files=files, method=method, k=k, q=q)
        self._header = MASTERHEADER.copy()
        self._header["FILTER"] = fits.getval(files[0], "FILTER")
        self._header["EXPTIME"] = "-"
        self._type = "flat"
        self._create()
        self._normalize()

    def _normalize(self):
        bad_pixel_mask = create_masterflat_bad_pixel_mask(self.data.copy())
        num = len(np.where(bad_pixel_mask == 0)[0])
        bad_pixel_mask[bad_pixel_mask == 0] = np.nan
        self.data *= bad_pixel_mask
        logger.info(f"removing {num} bad (Nan) pixels from file:")
        self.data = replace_nans(self.data.copy(), value_func="nearest")
        cen_val = np.nanmax(ndimage.median_filter(self.data.copy(), size=2))
        logger.info(f"normalizing masteflat by dividing through {cen_val}")
        self.data /= cen_val
