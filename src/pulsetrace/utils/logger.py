from __future__ import annotations

import logging

from pulsetrace.config.schema import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    handlers: list[logging.Handler] = []
    if config.file:
        handlers.append(logging.FileHandler(config.file))
    else:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=getattr(logging, config.level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("shap").setLevel(logging.WARNING)
