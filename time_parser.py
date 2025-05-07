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

    # Explicit calendar last week: Monday → Sunday of previous week
    if lower == "last week":
        # Compute this week's Monday
        this_week_monday = now - timedelta(days=now.weekday())
        # Shift back one week
        last_week_monday = this_week_monday - timedelta(days=7)
        last_week_sunday = last_week_monday + timedelta(days=6)
        return last_week_monday, last_week_sunday

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

    # Infer ranges
    if "hour" in lower:
        # Past hours up to dt
        delta_hours = int(''.join(filter(str.isdigit, lower))) or 1
        start = dt - timedelta(hours=delta_hours)
        end = dt
    elif "day" in lower or "yesterday" in lower or "today" in lower:
        start = dt - timedelta(days=1)
        end = dt
    elif "week" in lower:
        # For non-calendar references like "past week"
        start = dt - timedelta(days=7)
        end = dt
    else:
        # Default 24-hour window ending at dt
        start = dt - timedelta(days=1)
        end = dt

    return start, end
