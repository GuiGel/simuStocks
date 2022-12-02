import logging

__version__ = "0.1.0"


# Create a logger for the project
logger = logging.getLogger("simustocks")
if not logger.handlers:  # To ensure reload() doesn't add another one
    logger.addHandler(logging.NullHandler())
