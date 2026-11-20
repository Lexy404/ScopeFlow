from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.llm import ToolSpec
from app.tools.base import ToolError


class CurrentTime:
    spec = ToolSpec(
        name="current_time",
        description="Get the current date, time and weekday, optionally in an IANA timezone.",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone such as 'Asia/Jerusalem'. Defaults to UTC.",
                }
            },
        },
    )

    def __init__(self, now: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self._now = now  # injectable for tests

    async def run(self, args: dict[str, Any]) -> str:
        tz_name = str(args.get("timezone") or "UTC")
        try:
            tz = ZoneInfo(tz_name)
        except (ZoneInfoNotFoundError, ValueError) as e:
            raise ToolError(f"unknown timezone '{tz_name}'; use an IANA name") from e
        now = self._now().astimezone(tz)
        return f"{now:%A, %Y-%m-%d %H:%M:%S} ({tz_name}, UTC{now:%z})"
