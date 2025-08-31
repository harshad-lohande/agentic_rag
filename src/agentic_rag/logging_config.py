# src/agentic_rag/logging_config.py

import logging.config
import warnings
from pythonjsonlogger import jsonlogger
from agentic_rag.app.middlewares import REQUEST_ID_CTX_VAR


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    A custom JSON log formatter that adds the request_id from a context variable.
    """
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['request_id'] = REQUEST_ID_CTX_VAR.get()


# Suppress the specific Protobuf UserWarning from the gRPC libraries
warnings.filterwarnings("ignore", message="Protobuf gencode version.*")


def setup_logging():
    """
    Loads the logging configuration from the logging.ini file.
    This should be called at the start of any application entry point.
    """
    logging.config.fileConfig("logging.ini", disable_existing_loggers=False)


# Get the logger instance to be imported by other modules
logger = logging.getLogger("agentic_rag")