"""Parse Python log file message, in caf.toolkit format."""

import argparse
import datetime
import enum
import logging
import pathlib
import re
import warnings
from collections.abc import Generator
from typing import Annotated, Self

import pydantic

_NAME = pathlib.Path(__file__).stem if __name__ == "__main__" else __name__
LOG = logging.getLogger(_NAME)

LOG_PATTERN = re.compile(
    r"(?P<date>\d{2}-\d{2}-\d{4}\s\d{2}:\d{2}:\d{2})"  # date and time
    r"\s\[(?P<name>[\w\.]+)\s*\]"  # logger name
    r"\s\[(?P<level>\w+)\s*\]"  # log level
    r"(?P<message>.*?)"  # message
    r"(?=\d{2}-\d{2}-\d{4}\s\d{2}:\d{2}:\d{2}|$)",  # Finish at next date time
    flags=re.IGNORECASE | re.DOTALL,
)
_DATETIME_FORMAT = "%d-%m-%Y %H:%M:%S"


class LogLevel(enum.StrEnum):
    DEBUG = enum.auto()
    INFO = enum.auto()
    WARNING = enum.auto()
    ERROR = enum.auto()
    CRITICAL = enum.auto()

    @classmethod
    def _missing_(cls, value) -> Self:
        value = str(value)
        for member in cls:
            if value.lower().strip() == member.value:
                return member
        return None


class LogData(pydantic.BaseModel):
    date: Annotated[
        datetime.datetime,
        pydantic.BeforeValidator(lambda x: datetime.datetime.strptime(x, _DATETIME_FORMAT)),
    ]
    name: str
    level: LogLevel
    message: str

    def __str__(self) -> str:
        """Log format output."""
        return (
            f"{self.date.strftime(_DATETIME_FORMAT)}"
            f" [{self.name:40.40}] [{self.level:8.8}] {self.message}"
        )


class Format(enum.StrEnum):
    JSONL = enum.auto()
    LOG = enum.auto()


def parse_log(
    path: pathlib.Path, exclude_levels: set[LogLevel] | None = None
) -> Generator[LogData, None, None]:
    if exclude_levels is not None:
        LOG.info(
            "%s log levels excluded from output: %s",
            len(exclude_levels),
            ", ".join(i.value for i in exclude_levels),
        )

    with open(path, encoding="utf-8") as file:
        text = file.read()

    for matched in LOG_PATTERN.finditer(text):
        data = LogData(**matched.groupdict())
        if exclude_levels is not None and data.level in exclude_levels:
            continue
        yield data


def write_filtered(
    path: pathlib.Path,
    output_path: pathlib.Path | None = None,
    exclude_levels: set[LogLevel] | None = None,
    format_: Format = Format.JSONL,
) -> pathlib.Path:
    LOG.info("Converting log file to JSON lines: %s", path.resolve())
    if output_path is None:
        output_path = path.with_suffix(f".{format_}")
    if output_path.is_file():
        raise FileExistsError(f"output file already exists: {output_path}")
    if output_path.suffix != f".{format_.value}":
        warnings.warn(
            f"unexpected output suffix ({output_path.suffix}) for {format_.value} format",
            RuntimeWarning,
            stacklevel=2,
        )

    count = 0
    with open(output_path, "wt", encoding="utf-8") as file:
        for record in parse_log(path, exclude_levels):
            if format_ == Format.JSONL:
                file.write(record.model_dump_json() + "\n")
            elif format_ == Format.LOG:
                file.write(str(record))
            else:
                raise NotImplementedError(f"cannot output to {format_}")
            count += 1

    LOG.info("Written %s records to %s", count, output_path.resolve())


def _filepath(path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=_filepath, help="Path to log file")
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="Path to save output to, defaults to file.jsonl",
    )
    parser.add_argument(
        "-e",
        "--exclude_levels",
        type=LogLevel,
        nargs="+",
        help="Optional log levels to exclude from output, default includes all.",
    )
    parser.add_argument(
        "-f", "--format", type=Format, help="Format to write output to.", default=Format.LOG
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    write_filtered(args.file, args.output, args.exclude_levels, args.format)


if __name__ == "__main__":
    main()
