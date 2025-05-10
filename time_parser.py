# time_parser.py
"""
Utility to parse natural-language timeframes into concrete start/end datetimes.
"""
from datetime import datetime, timedelta
import dateparser
from zoneinfo import ZoneInfo
import re
from typing import Tuple, Optional


def extract_time_reference(text: str) -> Optional[str]:
    """
    Extract time reference from full query text.
    Returns the time reference if found, None otherwise.
    """
    # Common time patterns
    patterns = [
        r'(?:in|during|from|since|until|to|between)\s+(?:the\s+)?(?:last|past|next|this|previous)\s+\d+\s+(?:minute|hour|day|week|month|year)s?',
        r'(?:in|during|from|since|until|to|between)\s+(?:the\s+)?(?:last|past|next|this|previous)\s+(?:minute|hour|day|week|month|year)s?',
        r'(?:\d+\s+(?:minute|hour|day|week|month|year)s?\s+ago)',
        r'(?:yesterday|today|tomorrow)',
        r'(?:this|last|next)\s+(?:week|month|year)',
        r'(?:from|since|until|to|between)\s+\d{4}-\d{2}-\d{2}(?:\s+to\s+\d{4}-\d{2}-\d{2})?'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(0)
    return None


def extract_number(text: str) -> Optional[int]:
    """Extract a number from text."""
    match = re.search(r'\d+', text)
    return int(match.group(0)) if match else None


def get_period_start(period: str, number: int = 1) -> datetime:
    """Get the start of a time period."""
    now = datetime.now()
    if period == 'minute':
        return now - timedelta(minutes=number)
    elif period == 'hour':
        return now - timedelta(hours=number)
    elif period == 'day':
        return now - timedelta(days=number)
    elif period == 'week':
        return now - timedelta(weeks=number)
    elif period == 'month':
        return now - timedelta(days=30*number)
    elif period == 'year':
        return now - timedelta(days=365*number)
    return now


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
      - "past X days/hours/minutes"
      - "last X days/hours/minutes"
      - "X days/hours/minutes ago"
      - "yesterday", "today", "this week", "this month"
      - "YYYY-MM-DD to YYYY-MM-DD"
      - "between ... and ..."
      - "since X" (e.g., "since yesterday", "since last week")
      - "until X" (e.g., "until tomorrow", "until next week")
      - "from X to Y" (e.g., "from yesterday to today")
      - "during X" (e.g., "during last week", "during this month")

    Returns:
      (start_dt, end_dt)
    Raises:
      ValueError if parsing fails.
    """
    # Extract time reference from the full text
    time_ref = extract_time_reference(text)
    lower = time_ref.lower().strip() if time_ref else text.lower().strip()
    
    tzinfo = ZoneInfo(timezone)
    if now is None:
        now_dt = datetime.now(tzinfo)
    else:
        now_dt = now if now.tzinfo else now.replace(tzinfo=tzinfo)

    settings = {
        "TIMEZONE": timezone,
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "past",
    }

    # Helper function to extract numbers from text
    def extract_number(text: str) -> int:
        match = re.search(r'\d+', text)
        return int(match.group()) if match else 1

    # Helper function to get start of period
    def get_period_start(period: str) -> datetime:
        if period == "day":
            return now_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "week":
            return (now_dt - timedelta(days=now_dt.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "month":
            return now_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif period == "year":
            return now_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        return now_dt

    # Handle "since X" format
    if lower.startswith("since "):
        start = dateparser.parse(lower[6:], settings=settings)
        if start:
            return start, now_dt

    # Handle "until X" format
    if lower.startswith("until "):
        end = dateparser.parse(lower[6:], settings=settings)
        if end:
            return now_dt - timedelta(days=7), end  # Default to past week

    # Handle "during X" format
    if lower.startswith("during "):
        period = lower[7:].strip()
        if "last" in period:
            if "week" in period:
                start = get_period_start("week") - timedelta(weeks=1)
                end = get_period_start("week") - timedelta(seconds=1)
                return start, end
            elif "month" in period:
                start = get_period_start("month") - timedelta(days=30)
                end = get_period_start("month") - timedelta(seconds=1)
                return start, end
        elif "this" in period:
            if "week" in period:
                start = get_period_start("week")
                return start, now_dt
            elif "month" in period:
                start = get_period_start("month")
                return start, now_dt

    # Handle "from X to Y" format
    if " from " in lower and " to " in lower:
        parts = lower.split(" from ", 1)[1].split(" to ", 1)
        if len(parts) == 2:
            start = dateparser.parse(parts[0], settings=settings)
            end = dateparser.parse(parts[1], settings=settings)
            if start and end:
                return start, end

    # Calendar "last week": previous Monday → Sunday
    if "last week" in lower:
        this_week_start = get_period_start("week")
        last_week_start = this_week_start - timedelta(weeks=1)
        last_week_end = this_week_start - timedelta(seconds=1)
        return last_week_start, last_week_end

    # Handle "this week/month"
    if "this week" in lower:
        start = get_period_start("week")
        return start, now_dt
    if "this month" in lower:
        start = get_period_start("month")
        return start, now_dt

    # Handle "yesterday" and "today"
    if "yesterday" in lower:
        yesterday = now_dt - timedelta(days=1)
        start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        return start, end
    if "today" in lower:
        start = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        return start, end

    # Handle "X time ago" format
    ago_match = re.search(r'(\d+)\s*(day|hour|minute|week|month|year)s?\s+ago', lower)
    if ago_match:
        num = int(ago_match.group(1))
        unit = ago_match.group(2)
        if unit == "day":
            start = now_dt - timedelta(days=num)
        elif unit == "hour":
            start = now_dt - timedelta(hours=num)
        elif unit == "minute":
            start = now_dt - timedelta(minutes=num)
        elif unit == "week":
            start = now_dt - timedelta(weeks=num)
        elif unit == "month":
            start = now_dt - timedelta(days=num * 30)
        elif unit == "year":
            start = now_dt - timedelta(days=num * 365)
        return start, now_dt

    # Handle "past/last X time" format
    past_match = re.search(r'(?:past|last)\s+(\d+)\s*(day|hour|minute|week|month|year)s?', lower)
    if past_match:
        num = int(past_match.group(1))
        unit = past_match.group(2)
        if unit == "day":
            start = now_dt - timedelta(days=num)
        elif unit == "hour":
            start = now_dt - timedelta(hours=num)
        elif unit == "minute":
            start = now_dt - timedelta(minutes=num)
        elif unit == "week":
            start = now_dt - timedelta(weeks=num)
        elif unit == "month":
            start = now_dt - timedelta(days=num * 30)
        elif unit == "year":
            start = now_dt - timedelta(days=num * 365)
        return start, now_dt

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
        num = extract_number(lower)
        start = dt - timedelta(hours=num)
        end = dt
    elif any(keyword in lower for keyword in ["day", "yesterday", "today"]):
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif "week" in lower:
        num = extract_number(lower)
        start = dt - timedelta(weeks=num)
        end = dt
    elif "month" in lower:
        num = extract_number(lower)
        start = dt - timedelta(days=num * 30)
        end = dt
    elif "year" in lower:
        num = extract_number(lower)
        start = dt - timedelta(days=num * 365)
        end = dt
    else:
        # default to one day
        start = dt - timedelta(days=1)
        end = dt

    return start, end
