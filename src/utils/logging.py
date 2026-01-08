import sys
from loguru import logger

def setup_logging(level="INFO", show_time=True):
    """Configure loguru for the project.
    
    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
    show_time : bool
        Whether to show timestamps in the output.
    """
    # Remove default handler
    logger.remove()
    
    # Define format
    if show_time:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
    else:
        log_format = (
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
    # Add standardized handler to stderr
    logger.add(sys.stderr, format=log_format, level=level, colorize=True)
    
    return logger

# Default setup
setup_logging()
