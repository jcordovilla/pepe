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
    # Enhanced time reference patterns for comprehensive natural language support
    patterns = [
        # Basic relative periods
        r'past \d+ (?:days?|hours?|minutes?|weeks?|months?|years?)',
        r'last \d+ (?:days?|hours?|minutes?|weeks?|months?|years?)',
        r'previous \d+ (?:days?|hours?|minutes?|weeks?|months?|years?)',
        
        # Single periods
        r'past (?:week|month|year|hour|day)',
        r'last (?:week|month|year|hour|day)',
        r'previous (?:week|month|year|hour|day)',
        
        # Digest patterns - NEW
        r'(?:weekly|monthly|daily|quarterly|yearly) digest',
        r'digest (?:for|of) (?:the )?(?:week|month|day|quarter|year)',
        r'(?:week|month|day|quarter|year)ly summary',
        r'summary (?:for|of) (?:the )?(?:past|last) (?:week|month|day|quarter|year)',
        
        # Today/yesterday/time of day
        r'today',
        r'yesterday',
        r'this morning',
        r'this afternoon',
        r'this evening',
        r'tonight',
        r'this hour',
        
        # Time ago expressions  
        r'\d+ (?:hours?|minutes?|days?|weeks?|months?|years?) ago',
        
        # Month/season references
        r'(?:this|last|next) (?:month|quarter|year)',
        r'(?:january|february|march|april|may|june|july|august|september|october|november|december) \d{4}',
        r'last (?:january|february|march|april|may|june|july|august|september|october|november|december)',
        r'(?:this|last|next) (?:spring|summer|fall|autumn|winter)',
        
        # Flexible wording
        r'in the past (?:week|month|year|\d+ (?:days?|weeks?|months?))',
        r'over the last (?:week|month|year|\d+ (?:days?|weeks?|months?))',
        r'during the past (?:week|month|year|\d+ (?:days?|weeks?|months?))',
        r'within the last (?:week|month|year|\d+ (?:days?|weeks?|months?))',
        
        # Day of week references
        r'(?:last|this|next) (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
        r'since (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
        
        # Weekend references
        r'(?:last|this|next) weekend',
        
        # Date ranges
        r'from \d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}',
        r'between \d{4}-\d{2}-\d{2} and \d{4}-\d{2}-\d{2}',
        r'\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}',
        
        # Open-ended ranges
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

def get_last_calendar_week(now: datetime) -> Tuple[datetime, datetime]:
    """
    Get the last complete calendar week (Monday to Sunday).
    """
    # Get the current weekday (Monday=0, Sunday=6)
    current_weekday = now.weekday()
    
    # Calculate days back to the start of last week (Monday)
    days_to_last_monday = current_weekday + 7
    
    # Calculate start and end of last week
    last_monday = now - timedelta(days=days_to_last_monday)
    last_sunday = last_monday + timedelta(days=6)
    
    # Set times to start/end of day
    start_of_week = last_monday.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_week = last_sunday.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    return start_of_week, end_of_week


def get_day_of_week_date(day_name: str, reference: str, now: datetime) -> datetime:
    """
    Get the date for a specific day of the week relative to now.
    
    Args:
        day_name: Name of the day (monday, tuesday, etc.)
        reference: 'last', 'this', or 'next'
        now: Current datetime
        
    Returns:
        datetime for the specified day
    """
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    target_weekday = days.index(day_name.lower())
    current_weekday = now.weekday()
    
    if reference == 'last':
        # Find the most recent occurrence of this day
        days_back = (current_weekday - target_weekday) % 7
        if days_back == 0:  # If it's the same day, go back a week
            days_back = 7
        target_date = now - timedelta(days=days_back)
    elif reference == 'this':
        # Find this week's occurrence
        days_diff = target_weekday - current_weekday
        if days_diff < 0:  # Day already passed this week
            days_diff += 7
        target_date = now + timedelta(days=days_diff)
    elif reference == 'next':
        # Find next week's occurrence
        days_diff = target_weekday - current_weekday
        if days_diff <= 0:  # Include today as "next week"
            days_diff += 7
        target_date = now + timedelta(days=days_diff)
    else:
        raise ValueError(f"Invalid reference: {reference}")
    
    return target_date.replace(hour=0, minute=0, second=0, microsecond=0)


def get_month_boundaries(month_name: str, year: Optional[int] = None, now: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    """
    Get start and end dates for a specific month.
    
    Args:
        month_name: Name of the month
        year: Year (if None, uses current or previous year based on context)
        now: Current datetime for reference
        
    Returns:
        Tuple of (start_of_month, end_of_month)
    """
    if now is None:
        now = datetime.now(ZoneInfo("UTC"))
    
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    month_num = months[month_name.lower()]
    
    if year is None:
        year = now.year
        # If the month hasn't occurred yet this year, use last year
        if month_num > now.month:
            year -= 1
    
    # Start of month
    start_date = datetime(year, month_num, 1, tzinfo=ZoneInfo("UTC"))
    
    # End of month
    if month_num == 12:
        end_date = datetime(year + 1, 1, 1, tzinfo=ZoneInfo("UTC")) - timedelta(microseconds=1)
    else:
        end_date = datetime(year, month_num + 1, 1, tzinfo=ZoneInfo("UTC")) - timedelta(microseconds=1)
    
    return start_date, end_date


def get_time_of_day_boundaries(time_ref: str, now: datetime) -> Tuple[datetime, datetime]:
    """
    Get boundaries for time-of-day references like 'this morning', 'tonight', etc.
    
    Args:
        time_ref: Time reference string
        now: Current datetime
        
    Returns:
        Tuple of (start_time, end_time)
    """
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    if time_ref in ['today']:
        return today_start, now
    elif time_ref == 'yesterday':
        yesterday_start = today_start - timedelta(days=1)
        yesterday_end = today_start - timedelta(microseconds=1)
        return yesterday_start, yesterday_end
    elif time_ref == 'this morning':
        morning_start = today_start
        morning_end = today_start.replace(hour=12)
        return morning_start, min(morning_end, now)
    elif time_ref == 'this afternoon':
        afternoon_start = today_start.replace(hour=12)
        afternoon_end = today_start.replace(hour=18)
        return max(afternoon_start, today_start), min(afternoon_end, now)
    elif time_ref in ['this evening', 'tonight']:
        evening_start = today_start.replace(hour=18)
        evening_end = today_start.replace(hour=23, minute=59, second=59, microsecond=999999)
        return max(evening_start, today_start), min(evening_end, now)
    elif time_ref == 'this hour':
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        return hour_start, now
    else:
        raise ValueError(f"Unknown time reference: {time_ref}")


def parse_timeframe(query: str, now: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    """
    Parse comprehensive natural language timeframe from query string.
    Returns tuple of (start_datetime, end_datetime).
    Raises ValueError if no timeframe is specified.
    
    Supports:
    - Relative periods: "past 5 days", "last 2 weeks", "previous 3 months"
    - Time ago: "2 hours ago", "30 minutes ago"
    - Day references: "today", "yesterday", "last Monday"
    - Time of day: "this morning", "tonight"
    - Month references: "April 2025", "last April"
    - Flexible wording: "in the past week", "over the last month"
    - Date ranges: "2025-04-01 to 2025-04-07"
    - Open ranges: "since 2025-05-01", "until 2025-06-01"
    - Weekend references: "last weekend", "this weekend"
    """
    # Get current time in UTC (or use provided time for testing)
    if now is None:
        now = datetime.now(ZoneInfo("UTC"))
    elif now.tzinfo is None:
        # If provided time has no timezone, treat as UTC
        now = now.replace(tzinfo=ZoneInfo("UTC"))
    else:
        # Convert to UTC for consistency
        now = now.astimezone(ZoneInfo("UTC"))
    
    query_lower = query.lower().strip()
    
    # Comprehensive timeframe patterns with enhanced coverage
    patterns = {
        # Multiple unit relative periods (enhanced)
        r'past (\d+) days?': lambda m: (now - timedelta(days=int(m.group(1))), now),
        r'last (\d+) days?': lambda m: (now - timedelta(days=int(m.group(1))), now),
        r'previous (\d+) days?': lambda m: (now - timedelta(days=int(m.group(1))), now),
        r'past (\d+) weeks?': lambda m: (now - timedelta(weeks=int(m.group(1))), now),
        r'last (\d+) weeks?': lambda m: (now - timedelta(weeks=int(m.group(1))), now),
        r'previous (\d+) weeks?': lambda m: (now - timedelta(weeks=int(m.group(1))), now),
        r'past (\d+) months?': lambda m: (now - timedelta(days=30*int(m.group(1))), now),
        r'last (\d+) months?': lambda m: (now - timedelta(days=30*int(m.group(1))), now),
        r'previous (\d+) months?': lambda m: (now - timedelta(days=30*int(m.group(1))), now),
        r'past (\d+) years?': lambda m: (now - timedelta(days=365*int(m.group(1))), now),
        r'last (\d+) years?': lambda m: (now - timedelta(days=365*int(m.group(1))), now),
        r'previous (\d+) years?': lambda m: (now - timedelta(days=365*int(m.group(1))), now),
        r'past (\d+) hours?': lambda m: (now - timedelta(hours=int(m.group(1))), now),
        r'last (\d+) hours?': lambda m: (now - timedelta(hours=int(m.group(1))), now),
        r'past (\d+) minutes?': lambda m: (now - timedelta(minutes=int(m.group(1))), now),
        r'last (\d+) minutes?': lambda m: (now - timedelta(minutes=int(m.group(1))), now),
        
        # Time ago expressions
        r'(\d+) hours? ago': lambda m: (now - timedelta(hours=int(m.group(1))), now),
        r'(\d+) minutes? ago': lambda m: (now - timedelta(minutes=int(m.group(1))), now),
        r'(\d+) days? ago': lambda m: (now - timedelta(days=int(m.group(1))), now),
        r'(\d+) weeks? ago': lambda m: (now - timedelta(weeks=int(m.group(1))), now),
        r'(\d+) months? ago': lambda m: (now - timedelta(days=30*int(m.group(1))), now),
        r'(\d+) years? ago': lambda m: (now - timedelta(days=365*int(m.group(1))), now),
        
        # Single periods
        r'past week': lambda m: (now - timedelta(days=7), now),
        r'last week': lambda m: get_last_calendar_week(now),
        r'previous week': lambda m: (now - timedelta(days=7), now),
        r'past month': lambda m: (now - timedelta(days=30), now),
        r'last month': lambda m: (now - timedelta(days=30), now),
        r'previous month': lambda m: (now - timedelta(days=30), now),
        r'this month': lambda m: (now.replace(day=1, hour=0, minute=0, second=0, microsecond=0), now),
        r'past year': lambda m: (now - timedelta(days=365), now),
        r'last year': lambda m: (now - timedelta(days=365), now),
        r'previous year': lambda m: (now - timedelta(days=365), now),
        r'this year': lambda m: (now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0), now),
        r'past hour': lambda m: (now - timedelta(hours=1), now),
        r'last hour': lambda m: (now - timedelta(hours=1), now),
        r'this hour': lambda m: (now.replace(minute=0, second=0, microsecond=0), now),
        
        # Digest patterns - NEW
        r'weekly digest': lambda m: (now - timedelta(days=7), now),
        r'daily digest': lambda m: (now - timedelta(days=1), now),
        r'monthly digest': lambda m: (now - timedelta(days=30), now),
        r'quarterly digest': lambda m: (now - timedelta(days=90), now),
        r'yearly digest': lambda m: (now - timedelta(days=365), now),
        r'digest for (?:the )?week': lambda m: (now - timedelta(days=7), now),
        r'digest of (?:the )?week': lambda m: (now - timedelta(days=7), now),
        r'digest for (?:the )?month': lambda m: (now - timedelta(days=30), now),
        r'digest of (?:the )?month': lambda m: (now - timedelta(days=30), now),
        r'digest for (?:the )?day': lambda m: (now - timedelta(days=1), now),
        r'digest of (?:the )?day': lambda m: (now - timedelta(days=1), now),
        r'weekly summary': lambda m: (now - timedelta(days=7), now),
        r'daily summary': lambda m: (now - timedelta(days=1), now),
        r'monthly summary': lambda m: (now - timedelta(days=30), now),
        r'summary for (?:the )?(?:past|last) week': lambda m: (now - timedelta(days=7), now),
        r'summary of (?:the )?(?:past|last) week': lambda m: (now - timedelta(days=7), now),
        r'summary for (?:the )?(?:past|last) month': lambda m: (now - timedelta(days=30), now),
        r'summary of (?:the )?(?:past|last) month': lambda m: (now - timedelta(days=30), now),
        
        # Today/yesterday/time of day
        r'today': lambda m: get_time_of_day_boundaries('today', now),
        r'yesterday': lambda m: get_time_of_day_boundaries('yesterday', now),
        r'this morning': lambda m: get_time_of_day_boundaries('this morning', now),
        r'this afternoon': lambda m: get_time_of_day_boundaries('this afternoon', now),
        r'this evening': lambda m: get_time_of_day_boundaries('this evening', now),
        r'tonight': lambda m: get_time_of_day_boundaries('tonight', now),
        
        # Day of week references
        r'last (monday|tuesday|wednesday|thursday|friday|saturday|sunday)': lambda m: (
            get_day_of_week_date(m.group(1), 'last', now),
            get_day_of_week_date(m.group(1), 'last', now).replace(hour=23, minute=59, second=59, microsecond=999999)
        ),
        r'this (monday|tuesday|wednesday|thursday|friday|saturday|sunday)': lambda m: (
            get_day_of_week_date(m.group(1), 'this', now),
            get_day_of_week_date(m.group(1), 'this', now).replace(hour=23, minute=59, second=59, microsecond=999999)
        ),
        r'since (monday|tuesday|wednesday|thursday|friday|saturday|sunday)': lambda m: (
            get_day_of_week_date(m.group(1), 'last', now),
            now
        ),
        
        # Weekend references
        r'last weekend': lambda m: (
            get_day_of_week_date('saturday', 'last', now),
            get_day_of_week_date('sunday', 'last', now).replace(hour=23, minute=59, second=59, microsecond=999999)
        ),
        r'this weekend': lambda m: (
            get_day_of_week_date('saturday', 'this', now),
            get_day_of_week_date('sunday', 'this', now).replace(hour=23, minute=59, second=59, microsecond=999999)
        ),
        
        # Month references with year
        r'(january|february|march|april|may|june|july|august|september|october|november|december) (\d{4})': lambda m: get_month_boundaries(m.group(1), int(m.group(2)), now),
        r'last (january|february|march|april|may|june|july|august|september|october|november|december)': lambda m: get_month_boundaries(m.group(1), None, now),
        
        # Flexible wording patterns
        r'in the past (\d+) days?': lambda m: (now - timedelta(days=int(m.group(1))), now),
        r'in the past (\d+) weeks?': lambda m: (now - timedelta(weeks=int(m.group(1))), now),
        r'in the past (\d+) months?': lambda m: (now - timedelta(days=30*int(m.group(1))), now),
        r'in the past week': lambda m: (now - timedelta(days=7), now),
        r'in the past month': lambda m: (now - timedelta(days=30), now),
        r'in the past year': lambda m: (now - timedelta(days=365), now),
        
        r'over the last (\d+) days?': lambda m: (now - timedelta(days=int(m.group(1))), now),
        r'over the last (\d+) weeks?': lambda m: (now - timedelta(weeks=int(m.group(1))), now),
        r'over the last (\d+) months?': lambda m: (now - timedelta(days=30*int(m.group(1))), now),
        r'over the last week': lambda m: (now - timedelta(days=7), now),
        r'over the last month': lambda m: (now - timedelta(days=30), now),
        r'over the last year': lambda m: (now - timedelta(days=365), now),
        
        r'during the past (\d+) days?': lambda m: (now - timedelta(days=int(m.group(1))), now),
        r'during the past (\d+) weeks?': lambda m: (now - timedelta(weeks=int(m.group(1))), now),
        r'during the past (\d+) months?': lambda m: (now - timedelta(days=30*int(m.group(1))), now),
        r'during the past week': lambda m: (now - timedelta(days=7), now),
        r'during the past month': lambda m: (now - timedelta(days=30), now),
        r'during the past year': lambda m: (now - timedelta(days=365), now),
        
        r'within the last (\d+) days?': lambda m: (now - timedelta(days=int(m.group(1))), now),
        r'within the last (\d+) weeks?': lambda m: (now - timedelta(weeks=int(m.group(1))), now),
        r'within the last (\d+) months?': lambda m: (now - timedelta(days=30*int(m.group(1))), now),
        r'within the last week': lambda m: (now - timedelta(days=7), now),
        r'within the last month': lambda m: (now - timedelta(days=30), now),
        r'within the last year': lambda m: (now - timedelta(days=365), now),
        
        # Date ranges (existing patterns)
        r'(\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})': lambda m: (
            datetime.fromisoformat(m.group(1)).replace(tzinfo=ZoneInfo("UTC")),
            datetime.fromisoformat(m.group(2)).replace(tzinfo=ZoneInfo("UTC"))
        ),
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
    
    # Try to match each pattern (order matters for some patterns)
    for pattern, handler in patterns.items():
        match = re.search(pattern, query_lower)
        if match:
            try:
                start_dt, end_dt = handler(match)
                
                # Validate that start time is before end time
                if start_dt >= end_dt:
                    raise ValueError("End time must be after start time")
                
                logger.info(f"Parsed timeframe '{query}' -> {start_dt} to {end_dt}")
                return start_dt, end_dt
            except (ValueError, AttributeError) as e:
                logger.error(f"Failed to parse timeframe '{query}': {str(e)}")
                raise ValueError(f"Could not parse timeframe: {str(e)}")
    
    # If no pattern matches, raise an error instead of defaulting
    raise ValueError("No timeframe specified in query")
