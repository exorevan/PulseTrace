import logging
import typing as ty


def setup_logging(logging_config: dict[str, ty.Any]) -> None:
    level_str: str = ty.cast(str, logging_config.get("level", "INFO")).upper()

    level: int = getattr(logging, level_str, logging.INFO)
    log_file: str | None = logging_config.get("file", None)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=log_file if log_file else None,
    )
