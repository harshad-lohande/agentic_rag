# app/logging_config.py

import logging.config

def setup_logging():
    """
    Loads the logging configuration from the logging.ini file.
    This should be called at the start of any application entry point.
    """
    logging.config.fileConfig('logging.ini', disable_existing_loggers=False)

# Get the logger instance to be imported by other modules
logger = logging.getLogger("agentic_rag")