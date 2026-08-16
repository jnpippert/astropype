import os
from pathlib import Path
from multiprocessing import Pool, Queue
from logging.handlers import QueueListener
from astropy.io import fits
from tqdm import tqdm
from . import utilities as file_utils
from .logger import logger, Logger, configure_worker_logging

__all__ = ["init_pool"]

def single_process(files: list[Path], kwargs: dict) -> list[Path]:
    processed_files = []
    success = True
    files = file_utils.sort_files_by_obsdate(files)
    for file in (progressbar := tqdm(files, total=len(files))):
        progressbar.set_description(f"../{file.name} (processing)")
        try:
            new_filename = pool_func((file, kwargs))
            progressbar.set_description(f"../{file.name} (---deleted)")
            processed_files.append(new_filename)
        except:
            logger.exception(f"something went wrong processing '{file}'")
            success = False
    if success:
        logger.info("All files processed successfully.")
    return processed_files

def init_pool(files: list, kwargs: dict, nproc: int = None) -> list:
    if isinstance(files, (Path, str)):
        files = [files]
    files = [Path(file) if isinstance(file, str) else file for file in files]
    if nproc == 1:
        # No worker processes involved - the main process's logger already
        # writes straight to the real handlers, so no queue/listener needed.
        return single_process(files, kwargs)
    log_queue = Queue()
    listener = QueueListener(log_queue, *Logger.get_handlers(), respect_handler_level=True)
    listener.start()
    try:
        with Pool(processes=nproc, initializer=configure_worker_logging, initargs=(log_queue,)) as pool:
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
    finally:
        listener.stop()
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
    logger.info(f"\t{file}")
    new_filename = file.with_name(kwargs["prefix"] + file.name)
    data, header = kwargs["func"](file, kwargs)
    fits.writeto(new_filename, data, header, overwrite=True)

    try:
        os.remove(file) if kwargs["remove"] else None
        logger.info(f"\t\t... {file.name} removed")
    except FileNotFoundError:
        logger.info(f"No file or directory: ... {file.name}")

    return new_filename
