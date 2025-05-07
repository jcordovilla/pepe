# time_parser.py
"""
Utility to parse natural-language timeframes into concrete start/end datetimes.
"""
from datetime import datetime, timedelta
import dateparser
from zoneinfo import ZoneInfo


def parse_timeframe(text: str, timezone: str = "Europe/Madrid") -> tuple[datetime, datetime]:
    """
    Parse a natural-language timeframe into (start, end) datetimes in the given timezone.

    Supports:
      - "last week" (previous calendar week: Monday → Sunday)
      - "past X days/hours"
      - "YYYY-MM-DD to YYYY-MM-DD"
      - "between ... and ..."
      - "yesterday", "today"

    Returns:
      (start_dt, end_dt)
    Raises:
      ValueError if parsing fails.
    """
    lower = text.lower().strip()
    tzinfo = ZoneInfo(timezone)
    now = datetime.now(tzinfo)

    # Explicit calendar last week: previous Monday → Sunday
    if lower == "last week":
        # Calculate start of this week (Monday)
        this_week_start = (now - timedelta(days=now.weekday()))
        # Last week start is one week before
        last_week_start = this_week_start - timedelta(weeks=1)
        # Last week end is day before this week's start (Sunday)
        last_week_end = this_week_start - timedelta(days=1)
        # Normalize times
        start = last_week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = last_week_end.replace(hour=23, minute=59, second=59, microsecond=999999)
        return start, end

    settings = {
        "TIMEZONE": timezone,
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "past",
    }

    # Try explicit range separators first
    for sep in [" to ", " and ", "–", "-"]:
        if sep in text:
            start_str, end_str = text.split(sep, 1)
            start = dateparser.parse(start_str, settings=settings)
            end = dateparser.parse(end_str, settings=settings)
            if start and end:
                return start, end

    # Single-point parse and infer range
    dt = dateparser.parse(text, settings=settings)
    if not dt:
        raise ValueError(f"Could not parse timeframe: '{text}'")

    # Infer ranges for non-explicit calendar weeks
    if "hour" in lower:
        delta_hours = int(''.join(filter(str.isdigit, lower))) or 1
        start = dt - timedelta(hours=delta_hours)
        end = dt
    elif "day" in lower or "yesterday" in lower or "today" in lower:
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif "week" in lower:
        # Past week as last 7 days ending at dt
        start = dt - timedelta(days=7)
        end = dt
    else:
        # Default 24-hour window ending at dt
        start = dt - timedelta(days=1)
        end = dt

    return start, end
