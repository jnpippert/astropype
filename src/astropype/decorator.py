import time
from functools import wraps

__all__ = ["timeit"]


def timeit(func=None):
    def decorator_timeit(func):
        @wraps(func)
        def wrapper_timeit(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            print(f"\tfinished after {round(end_time - start_time,2)} seconds")
            return result

        return wrapper_timeit

    if func is None:  # @timeit() -> @timeit
        return decorator_timeit

    return decorator_timeit(func)
