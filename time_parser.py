# time_parser.py

from datetime import datetime, timedelta
import dateparser

def parse_timeframe(text: str, timezone: str = "Europe/Madrid") -> tuple[datetime, datetime]:
    """
    Parse a natural-language timeframe into (start, end) datetimes in the given timezone.

    Supports expressions like:
      - "last week"
      - "past 2 days"
      - "April 1 to April 7"
      - "between April 1 and April 7"
      - "yesterday", "today"

    Returns:
      (start_dt, end_dt)
    """
    settings = {
        "TIMEZONE": timezone,
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "past",
    }

    # Try parsing “to”/“and” ranges first
    for sep in [" to ", " and ", "–", "-"]:
        if sep in text:
            parts = text.split(sep, 1)
            start = dateparser.parse(parts[0], settings=settings)
            end = dateparser.parse(parts[1], settings=settings)
            if start and end:
                return start, end

    # Otherwise, parse a single reference point, then infer range
    dt = dateparser.parse(text, settings=settings)
    if not dt:
        raise ValueError(f"Could not parse timeframe: '{text}'")

    # Interpret ranges
    lower = text.lower().strip()
    if "week" in lower:
        # last week → Monday through Sunday of the previous week
        # dateparser gives a point; we'll take 7 days back
        end = dt
        start = dt - timedelta(days=7)
    elif "day" in lower or "yesterday" in lower or "today" in lower:
        end = dt
        start = dt - timedelta(days=1)
    else:
        # Default to a 24-hour window ending at dt
        end = dt
        start = dt - timedelta(days=1)

    return start, end
