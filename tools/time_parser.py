# time_parser.py
"""
Utility to parse natural-language timeframes into concrete start/end datetimes.
"""
from datetime import datetime, timedelta, timezone
import dateparser
from zoneinfo import ZoneInfo
import re
from typing import Tuple, Optional, Dict, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_time_reference(query: str) -> Optional[str]:
    """
    Extract time reference from query string.
    Returns the matched time reference or None.
    """
    # Common time reference patterns
    patterns = [
        r'past \d+ days?',
        r'last \d+ days?',
        r'previous \d+ days?',
        r'past week',
        r'last week',
        r'previous week',
        r'past month',
        r'last month',
        r'previous month',
        r'past year',
        r'last year',
        r'previous year',
        r'from \d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}',
        r'between \d{4}-\d{2}-\d{2} and \d{4}-\d{2}-\d{2}',
        r'from \d{4}-\d{2}-\d{2}',
        r'since \d{4}-\d{2}-\d{2}',
        r'until \d{4}-\d{2}-\d{2}',
        r'before \d{4}-\d{2}-\d{2}'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(0)
    
    return None

def extract_channel_reference(query: str) -> Optional[str]:
    """
    Extract channel reference from query string.
    Returns the channel name or None.
    """
    # Channel reference patterns
    patterns = [
        r'in #?([a-zA-Z0-9-]+)',
        r'from #?([a-zA-Z0-9-]+)',
        r'in channel #?([a-zA-Z0-9-]+)',
        r'from channel #?([a-zA-Z0-9-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1)
    
    return None

def extract_content_reference(query: str) -> str:
    """
    Extract content reference from query string.
    Returns the remaining keywords after removing time and channel references.
    """
    # Remove time reference
    time_ref = extract_time_reference(query)
    if time_ref:
        query = query.replace(time_ref, '')
    
    # Remove channel reference
    channel_ref = extract_channel_reference(query)
    if channel_ref:
        query = query.replace(f'in #{channel_ref}', '')
        query = query.replace(f'from #{channel_ref}', '')
    
    # Remove common query words
    common_words = ['show', 'me', 'messages', 'about', 'related', 'to', 'from', 'in', 'the']
    words = query.split()
    filtered_words = [w for w in words if w.lower() not in common_words]
    
    return ' '.join(filtered_words).strip()

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

def parse_timeframe(query: str) -> Tuple[datetime, datetime]:
    """
    Parse natural language timeframe from query string.
    Returns tuple of (start_datetime, end_datetime).
    Raises ValueError if no timeframe is specified.
    """
    # Get current time in UTC
    now = datetime.now(ZoneInfo("UTC"))
    
    # Common timeframe patterns
    patterns = {
        r'past (\d+) days?': lambda m: (now - timedelta(days=int(m.group(1))), now),
        r'last (\d+) days?': lambda m: (now - timedelta(days=int(m.group(1))), now),
        r'previous (\d+) days?': lambda m: (now - timedelta(days=int(m.group(1))), now),
        r'past week': lambda m: (now - timedelta(days=7), now),
        r'last week': lambda m: (now - timedelta(days=7), now),
        r'previous week': lambda m: (now - timedelta(days=7), now),
        r'past month': lambda m: (now - timedelta(days=30), now),
        r'last month': lambda m: (now - timedelta(days=30), now),
        r'previous month': lambda m: (now - timedelta(days=30), now),
        r'past year': lambda m: (now - timedelta(days=365), now),
        r'last year': lambda m: (now - timedelta(days=365), now),
        r'previous year': lambda m: (now - timedelta(days=365), now),
        r'from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})': lambda m: (
            datetime.fromisoformat(m.group(1)).replace(tzinfo=ZoneInfo("UTC")),
            datetime.fromisoformat(m.group(2)).replace(tzinfo=ZoneInfo("UTC"))
        ),
        r'between (\d{4}-\d{2}-\d{2}) and (\d{4}-\d{2}-\d{2})': lambda m: (
            datetime.fromisoformat(m.group(1)).replace(tzinfo=ZoneInfo("UTC")),
            datetime.fromisoformat(m.group(2)).replace(tzinfo=ZoneInfo("UTC"))
        ),
        r'from (\d{4}-\d{2}-\d{2})': lambda m: (
            datetime.fromisoformat(m.group(1)).replace(tzinfo=ZoneInfo("UTC")),
            now
        ),
        r'since (\d{4}-\d{2}-\d{2})': lambda m: (
            datetime.fromisoformat(m.group(1)).replace(tzinfo=ZoneInfo("UTC")),
            now
        ),
        r'until (\d{4}-\d{2}-\d{2})': lambda m: (
            now - timedelta(days=30),  # Default to last 30 days if only end date specified
            datetime.fromisoformat(m.group(1)).replace(tzinfo=ZoneInfo("UTC"))
        ),
        r'before (\d{4}-\d{2}-\d{2})': lambda m: (
            now - timedelta(days=30),  # Default to last 30 days if only end date specified
            datetime.fromisoformat(m.group(1)).replace(tzinfo=ZoneInfo("UTC"))
        )
    }
    
    # Try to match each pattern
    for pattern, handler in patterns.items():
        match = re.search(pattern, query.lower())
        if match:
            try:
                start_dt, end_dt = handler(match)
                return start_dt, end_dt
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Could not parse timeframe: {str(e)}")
    
    # If no pattern matches, raise an error instead of defaulting
    raise ValueError("No timeframe specified in query")
