import logging
import multiprocessing
from datetime import datetime
from logging.handlers import QueueHandler


class LevelFormatter(logging.Formatter):
    def format(self, record):
        record.levelname = f"{record.levelname:<7}"  # 7 - For longest level WARNING
        return super().format(record)


class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls, *args, **kwargs)
            cls.logger = None
            cls._instance._setup()
        return cls._instance

    def _setup(self):
        self.logger = logging.getLogger("pipeline")
        self.logger.setLevel(logging.DEBUG)

        if multiprocessing.parent_process() is not None:
            # Running inside a multiprocessing worker: don't open our own
            # console/file handlers here. init_pool()'s worker initializer
            # calls configure_worker_logging() instead, which attaches a
            # QueueHandler so records get forwarded to the main process and
            # land in the one logfile it already opened.
            return

        # Create handlers
        console_handler = logging.StreamHandler()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # delay=True: don't open the file until the first record is actually
        # written through this handler. Combined with configure_worker_logging()
        # stripping this handler before a worker ever logs anything, this is
        # what actually guarantees workers never create their own logfile -
        # multiprocessing.parent_process() above can read as None too early
        # in a worker's bootstrap (its args are unpickled, importing this
        # module, before multiprocessing finishes setting up that child's own
        # bookkeeping), so it's a best-effort skip, not a hard guarantee.
        file_handler = logging.FileHandler(f"{timestamp}.log", encoding="utf-8", delay=True)

        # Set levels for handlers
        console_handler.setLevel(logging.INFO)
        file_handler.setLevel(logging.DEBUG)

        # Create formatters and add them to handlers
        formatter = LevelFormatter(
            fmt="[%(levelname)s] %(asctime)s : %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    @staticmethod
    def get_logger():
        return Logger()._instance.logger

    @staticmethod
    def get_handlers():
        return list(Logger()._instance.logger.handlers)


def configure_worker_logging(queue):
    """
    Reconfigures the shared 'pipeline' logger inside a multiprocessing
    worker process to forward every record to ``queue`` instead of writing
    directly. Used as the ``initializer`` for ``multiprocessing.Pool`` in
    ``pool.py`` so every worker ends up logging to the same console/file
    the main process already opened, instead of each worker opening its
    own separate logfile.
    """
    worker_logger = Logger().logger
    for handler in list(worker_logger.handlers):
        worker_logger.removeHandler(handler)
    worker_logger.addHandler(QueueHandler(queue))


logger_instance = Logger()
logger = logger_instance.get_logger()
