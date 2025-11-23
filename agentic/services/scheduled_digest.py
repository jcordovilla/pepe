"""
Scheduled Digest Service

Provides automated digest generation and posting to Discord channels.
Uses APScheduler for scheduling if available, otherwise provides manual trigger.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ScheduledDigestService:
    """
    Automated digest generation and posting service.

    Features:
    - Weekly/monthly automated digests
    - Configurable posting channel
    - Optional scheduling via APScheduler
    - Manual trigger support
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the scheduled digest service.

        Args:
            config: Configuration options
        """
        self.config = config or {}

        # Configuration
        self.enabled = self.config.get("enabled", True)
        self.weekly_day = self.config.get("weekly_day", "monday")  # Day of week
        self.weekly_hour = self.config.get("weekly_hour", 9)  # Hour (24h format)
        self.monthly_day = self.config.get("monthly_day", 1)  # Day of month
        self.digest_channel_id = self.config.get("digest_channel_id") or os.getenv("DIGEST_CHANNEL_ID")

        # Components
        self._scheduler = None
        self._discord_client = None
        self._analytics_service = None

        logger.info(f"ScheduledDigestService initialized (enabled: {self.enabled})")

    @property
    def analytics_service(self):
        """Lazy-load analytics service."""
        if self._analytics_service is None:
            from .analytics_service import get_analytics_service
            self._analytics_service = get_analytics_service()
        return self._analytics_service

    def set_discord_client(self, client):
        """Set the Discord client for posting."""
        self._discord_client = client
        logger.info("Discord client set for scheduled digests")

    async def start_scheduler(self):
        """
        Start the automated scheduler.

        Attempts to use APScheduler if available, otherwise logs instructions.
        """
        if not self.enabled:
            logger.info("Scheduled digests are disabled")
            return

        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger

            self._scheduler = AsyncIOScheduler()

            # Weekly digest - every Monday at configured hour
            self._scheduler.add_job(
                self._post_weekly_digest,
                CronTrigger(
                    day_of_week=self._day_to_cron(self.weekly_day),
                    hour=self.weekly_hour,
                    minute=0
                ),
                id="weekly_digest",
                name="Weekly Community Digest"
            )

            # Monthly digest - 1st of every month
            self._scheduler.add_job(
                self._post_monthly_digest,
                CronTrigger(
                    day=self.monthly_day,
                    hour=self.weekly_hour,
                    minute=30
                ),
                id="monthly_digest",
                name="Monthly Community Digest"
            )

            self._scheduler.start()
            logger.info(f"Scheduler started: Weekly on {self.weekly_day} at {self.weekly_hour}:00")

        except ImportError:
            logger.warning(
                "APScheduler not available. Install with: pip install apscheduler\n"
                "Manual digest triggers still available via /digest command or trigger_digest()"
            )
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")

    async def stop_scheduler(self):
        """Stop the scheduler."""
        if self._scheduler:
            self._scheduler.shutdown()
            logger.info("Scheduler stopped")

    def _day_to_cron(self, day: str) -> str:
        """Convert day name to cron format."""
        days = {
            "monday": "mon",
            "tuesday": "tue",
            "wednesday": "wed",
            "thursday": "thu",
            "friday": "fri",
            "saturday": "sat",
            "sunday": "sun"
        }
        return days.get(day.lower(), "mon")

    async def _post_weekly_digest(self):
        """Generate and post weekly digest."""
        try:
            logger.info("Generating scheduled weekly digest...")

            result = await self.analytics_service.get_community_digest(period="weekly")

            if result.get("success"):
                await self._post_to_channel(
                    title="Weekly Community Digest",
                    content=result.get("digest", ""),
                    metrics=result.get("metrics", {}),
                    color=0xFFD700  # Gold
                )
                logger.info("Weekly digest posted successfully")
            else:
                logger.error(f"Failed to generate weekly digest: {result.get('error')}")

        except Exception as e:
            logger.error(f"Error posting weekly digest: {e}")

    async def _post_monthly_digest(self):
        """Generate and post monthly digest."""
        try:
            logger.info("Generating scheduled monthly digest...")

            result = await self.analytics_service.get_community_digest(period="monthly")

            if result.get("success"):
                await self._post_to_channel(
                    title="Monthly Community Digest",
                    content=result.get("digest", ""),
                    metrics=result.get("metrics", {}),
                    color=0x9B59B6  # Purple
                )
                logger.info("Monthly digest posted successfully")
            else:
                logger.error(f"Failed to generate monthly digest: {result.get('error')}")

        except Exception as e:
            logger.error(f"Error posting monthly digest: {e}")

    async def _post_to_channel(
        self,
        title: str,
        content: str,
        metrics: Dict[str, Any],
        color: int
    ):
        """Post digest to configured Discord channel."""
        if not self._discord_client:
            logger.warning("No Discord client set, cannot post digest")
            return

        if not self.digest_channel_id:
            logger.warning("No digest channel configured (set DIGEST_CHANNEL_ID)")
            return

        try:
            import discord

            channel = self._discord_client.get_channel(int(self.digest_channel_id))
            if not channel:
                channel = await self._discord_client.fetch_channel(int(self.digest_channel_id))

            if not channel:
                logger.error(f"Could not find channel {self.digest_channel_id}")
                return

            # Create embed
            embed = discord.Embed(
                title=title,
                description=content[:4000],
                color=discord.Color(color),
                timestamp=datetime.utcnow()
            )

            # Add metrics
            embed.add_field(
                name="Overview",
                value=f"Messages: {metrics.get('total_messages', 0):,}\n"
                      f"Channels: {metrics.get('unique_channels', 0)}\n"
                      f"Members: {metrics.get('unique_contributors', 0)}",
                inline=True
            )

            # Top contributors
            top_contributors = list(metrics.get("top_contributors", {}).items())[:5]
            if top_contributors:
                contrib_text = "\n".join([f"{name}: {count}" for name, count in top_contributors])
                embed.add_field(
                    name="Top Contributors",
                    value=contrib_text,
                    inline=True
                )

            embed.set_footer(text="Automated digest by Pepe Bot")

            await channel.send(embed=embed)
            logger.info(f"Digest posted to channel {channel.name}")

        except Exception as e:
            logger.error(f"Error posting to Discord channel: {e}")

    async def trigger_digest(
        self,
        period: str = "weekly",
        channel_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Manually trigger a digest generation and posting.

        Args:
            period: "daily", "weekly", or "monthly"
            channel_id: Optional override for target channel

        Returns:
            Result of the digest generation
        """
        try:
            logger.info(f"Manually triggering {period} digest...")

            result = await self.analytics_service.get_community_digest(period=period)

            if result.get("success"):
                original_channel = self.digest_channel_id
                if channel_id:
                    self.digest_channel_id = channel_id

                await self._post_to_channel(
                    title=f"{period.capitalize()} Community Digest",
                    content=result.get("digest", ""),
                    metrics=result.get("metrics", {}),
                    color=0xFFD700 if period == "weekly" else 0x9B59B6
                )

                self.digest_channel_id = original_channel

                return {"success": True, "message": f"{period.capitalize()} digest posted"}
            else:
                return result

        except Exception as e:
            logger.error(f"Error in manual digest trigger: {e}")
            return {"success": False, "error": str(e)}

    def get_schedule_info(self) -> Dict[str, Any]:
        """Get information about scheduled tasks."""
        info = {
            "enabled": self.enabled,
            "weekly": {
                "day": self.weekly_day,
                "hour": self.weekly_hour
            },
            "monthly": {
                "day": self.monthly_day,
                "hour": self.weekly_hour
            },
            "digest_channel_id": self.digest_channel_id,
            "scheduler_running": self._scheduler is not None and self._scheduler.running if self._scheduler else False
        }

        if self._scheduler and self._scheduler.running:
            jobs = self._scheduler.get_jobs()
            info["scheduled_jobs"] = [
                {"id": job.id, "name": job.name, "next_run": str(job.next_run_time)}
                for job in jobs
            ]

        return info


# Global instance
_scheduled_digest = None

def get_scheduled_digest_service(config: Optional[Dict[str, Any]] = None) -> ScheduledDigestService:
    """Get the global scheduled digest service instance."""
    global _scheduled_digest
    if _scheduled_digest is None:
        _scheduled_digest = ScheduledDigestService(config)
    return _scheduled_digest
