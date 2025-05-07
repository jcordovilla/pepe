# tests/test_time_parser.py
import pytest
from datetime import date, datetime
from zoneinfo import ZoneInfo
from time_parser import parse_timeframe

def test_explicit_range():
    start, end = parse_timeframe("2025-04-01 to 2025-04-07")
    assert start.date() == date(2025, 4, 1)
    assert end.date()   == date(2025, 4, 7)

def test_last_week_calendar():
    # Given today is May 7, 2025 in Europe/Madrid, last calendar week
    tz = ZoneInfo("Europe/Madrid")
    fake_now = datetime(2025, 5, 7, 12, 0, tzinfo=tz)
    start, end = parse_timeframe("last week", now=fake_now)
    # Last week: Monday 2025-04-28 through Sunday 2025-05-04
    assert start.date() == date(2025, 4, 28)
    assert end.date()   == date(2025, 5, 4)

def test_past_week_default():
    # Non-calendar week phrasing falls back to a 7-day window ending at 'now'
    tz = ZoneInfo("Europe/Madrid")
    fake_now = datetime(2025, 5, 7, 12, 0, tzinfo=tz)
    start, end = parse_timeframe("past week", now=fake_now)
    assert (end - start).days == 7
    assert end.date() == fake_now.date()
