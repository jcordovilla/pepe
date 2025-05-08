# utils/logger.py
# This file contains the setup for logging in the application.
import logging

def setup_logging():
    """
    Configure the root logger for the entire application.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
