import logging
import os

# Environment variable to set the application logger(s) in debug mode during runtime
LANGCHAIN_AWS_DEBUG: str = os.environ.get("LANGCHAIN_AWS_DEBUG", "false")
__DEBUG: bool = LANGCHAIN_AWS_DEBUG.lower() in ["true", "1"]

# Flag for root debug logger to set the root debug logger in debug mode during runtime
# Root debug logger will print boto3 as well as application debug logs if set to true
# This flag will be set to true if LANGCHAIN_AWS_DEBUG = LANGCHAIN_AWS_DEBUG_ROOT = true
__LANGCHAIN_AWS_DEBUG_ROOT: str = os.environ.get("LANGCHAIN_AWS_DEBUG_ROOT", "false")
__ROOT_DEBUG: bool = __DEBUG if __LANGCHAIN_AWS_DEBUG_ROOT.lower() \
    in ["true", "1"] else False

# If application level debug flag is set, set the default logging level to DEBUG 
# else ERROR
DEFAULT_LOG_LEVEL: int = logging.DEBUG if __DEBUG else logging.ERROR

#  Checking if we have a log file handler available, if not, setting default handler 
#  to None
DEFAULT_LOG_FILE: str = os.environ.get("LANGCHAIN_AWS_LOG_OUTPUT")
if DEFAULT_LOG_FILE:
    DEFAULT_LOG_HANDLER: logging.Handler = logging.FileHandler(DEFAULT_LOG_FILE, "a")
else:
    DEFAULT_LOG_HANDLER: logging.Handler = None

# If application debug mode set then include filename and line numbers in the log format
if __DEBUG:
    DEFAULT_LOG_FORMAT: str = (
        "%(asctime)s %(levelname)s | [%(filename)s:%(lineno)s] | %(name)s - %(message)s"
    )
else:
    DEFAULT_LOG_FORMAT: str = "%(asctime)s %(levelname)s | %(name)s - %(message)s"

is_colored_logs: bool = True
# This block is to set a log formatter. colorama is used to print colored logs
try:
    import colorama
    import coloredlogs

    colorama.init()

    # If a DEFAULT_LOG_HANDLER is set in the LANGCHAIN_AWS_LOG_OUTPUT env var, skip
    # colored logs format
    if DEFAULT_LOG_HANDLER:
        DEFAULT_LOG_FORMATTER: logging.Formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    else:
        DEFAULT_LOG_FORMATTER: logging.Formatter = coloredlogs.ColoredFormatter(
            DEFAULT_LOG_FORMAT
        )
# If colorama is not present, use the logging lib formatter without colored logs
except ImportError:
    colorama = None
    coloredlogs = None
    is_colored_logs = False
    DEFAULT_LOG_FORMATTER: logging.Formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

def get_logger(
    logger_name: str = None,
    log_handler: logging.Handler = DEFAULT_LOG_HANDLER,
    log_formatter: logging.Formatter = DEFAULT_LOG_FORMATTER,
    log_level: int = DEFAULT_LOG_LEVEL,
) -> logging.Logger:
    """
    Creates a logger with the passed logger_name at module level or function level

    Args:
        logger_name: Optional arg. Module/function name for the logger object. If the
        same logger_name is passed again in the function, the same logger object will
        be returned for the __LOGGER_CACHE
        log_handler: Optional arg. None by default. If default is None, sys.out is set
        as the default for a logger that has not been initialized (i.e. logger object
        not in __LOGGER_CACHE). Only one handler is supported per logger object in 
        this implementation
        log_formatter: Optional arg. Default format is determined by __DEBUG flag. If
        set, then include file and lines info. colorama is being used for coleredLogs.
        colored logs are only displayed in standard streams. Logging into files will
        display regular log format
        log_level: Optional arg. Log level to be set. default value will be 
        DEFAULT_LOG_LEVEL. The variable is set as DEBUG if LANGCHAIN_AWS_DEBUG 
        environment variable is True else it is set as INFO
        
    
    If logger_name is None or if the __DEBUG_ROOT flag is set, by default we want
    to initialize the root logger
    """

    if not logger_name or __ROOT_DEBUG:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(logger_name)

    # If log handler is None, set default stream as std.out
    if not log_handler:
        log_handler = logging.StreamHandler()

    # add formatter to handler
    log_handler.setFormatter(log_formatter)
    
    # Making sure only a single handler is present even if we have multiple
    # initializations with the same logger name
    if not len(logger.handlers):
        logger.addHandler(log_handler)
    
    logger.setLevel(log_level)
    if not is_colored_logs:
        logger.warning("Colored logs are not available while writing to a log file."
                       "Try importing colorama and coloredlogs before writing to"
                       " std out.")
    return logger

