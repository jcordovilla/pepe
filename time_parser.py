# time_parser.py
"""
Utility to parse natural-language timeframes into concrete start/end datetimes.
"""
from datetime import datetime, timedelta
import dateparser
from zoneinfo import ZoneInfo


def parse_timeframe(
    text: str,
    timezone: str = "Europe/Madrid",
    now: datetime = None
) -> tuple[datetime, datetime]:
    """
    Parse a natural-language timeframe into (start, end) datetimes in the given timezone.

    Supports:
      - "last week" (previous calendar week: Monday → Sunday)
      - "past week" (rolling 7-day window)
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
    if now is None:
        now_dt = datetime.now(tzinfo)
    else:
        now_dt = now if now.tzinfo else now.replace(tzinfo=tzinfo)

    # Calendar "last week": previous Monday → Sunday
    if "last week" in lower:
        this_week_start = (
            now_dt - timedelta(days=now_dt.weekday())
        ).replace(hour=0, minute=0, second=0, microsecond=0)
        last_week_start = this_week_start - timedelta(weeks=1)
        last_week_end = this_week_start - timedelta(seconds=1)
        return last_week_start, last_week_end

    # Rolling 7-day window
    if "past week" in lower:
        start = now_dt - timedelta(days=7)
        end = now_dt
        return start, end

    settings = {
        "TIMEZONE": timezone,
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "past",
    }

    # Explicit range separators
    for sep in [" to ", " and ", "–", "-"]:
        if sep in text:
            start_str, end_str = text.split(sep, 1)
            start = dateparser.parse(start_str, settings=settings)
            end = dateparser.parse(end_str, settings=settings)
            if start and end:
                return start, end

    # Fallback single date parse
    dt = dateparser.parse(text, settings=settings)
    if not dt:
        raise ValueError(f"Could not parse timeframe: '{text}'")

    # Infer range based on keywords
    if any(keyword in lower for keyword in ["hour", "hours"]):
        # e.g. "past 5 hours"
        num = int(''.join(filter(str.isdigit, lower))) or 1
        start = dt - timedelta(hours=num)
        end = dt
    elif any(keyword in lower for keyword in ["day", "yesterday", "today"]):
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif "week" in lower:
        # e.g. "2 weeks ago"
        num = int(''.join(filter(str.isdigit, lower))) if any(c.isdigit() for c in lower) else 1
        start = dt - timedelta(weeks=num)
        end = dt
    else:
        # default to one day
        start = dt - timedelta(days=1)
        end = dt

    return start, end
