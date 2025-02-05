import logging


def setup_logging(logging_config):
    level_str = logging_config.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    log_file = logging_config.get("file", None)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=log_file if log_file else None,
    )
