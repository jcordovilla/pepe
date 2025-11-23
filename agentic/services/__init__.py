"""
Modernized Services Package
Unified architecture with battle-tested patterns
"""

from .discord_service import DiscordMessageService
from .content_processor import ContentProcessingService
from .sync_service import DataSynchronizationService
from .unified_data_manager import UnifiedDataManager
from .analytics_service import AnalyticsService, get_analytics_service
from .report_generator import ReportGenerator
from .scheduled_digest import ScheduledDigestService, get_scheduled_digest_service

__all__ = [
    "DiscordMessageService",
    "ContentProcessingService",
    "DataSynchronizationService",
    "UnifiedDataManager",
    "AnalyticsService",
    "get_analytics_service",
    "ReportGenerator",
    "ScheduledDigestService",
    "get_scheduled_digest_service"
]
