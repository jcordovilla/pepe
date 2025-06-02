"""
Performance Monitor

Real-time performance monitoring and alerting system for the agentic framework.
Tracks system health, response times, error rates, and resource usage.
"""

import asyncio
import json
import psutil
import os
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import threading
import time

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of performance metrics"""
    RESPONSE_TIME = "response_time"
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    DISK_USAGE = "disk_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    ACTIVE_USERS = "active_users"
    QUERY_VOLUME = "query_volume"


@dataclass
class PerformanceAlert:
    """Performance alert data structure"""
    level: AlertLevel
    metric: MetricType
    current_value: float
    threshold: float
    message: str
    timestamp: datetime
    affected_components: List[str]
    suggested_actions: List[str]


@dataclass
class SystemMetrics:
    """System performance metrics snapshot"""
    timestamp: datetime
    response_time: float
    success_rate: float
    error_rate: float
    memory_usage: float
    cpu_usage: float
    disk_usage: float
    cache_hit_rate: float
    active_users: int
    query_volume: int
    system_load: float


class PerformanceMonitor:
    """
    Real-time performance monitoring system.
    
    Features:
    - Continuous system monitoring
    - Customizable thresholds and alerts
    - Performance trend analysis
    - Automatic health checks
    - Resource usage tracking
    - Integration with query repository
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitoring_enabled = config.get("monitoring_enabled", True)
        self.monitoring_interval = config.get("monitoring_interval", 30)  # seconds
        self.alert_thresholds = config.get("alert_thresholds", self._default_thresholds())
        self.alert_callbacks = []
        
        # Performance data storage
        self.metrics_history: List[SystemMetrics] = []
        self.max_history_size = config.get("max_history_size", 1000)
        self.alerts_history: List[PerformanceAlert] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.last_metrics: Optional[SystemMetrics] = None
        
        # Component references (set by parent system)
        self.query_repository = None
        self.vector_store = None
        self.cache = None
        
        logger.info("Performance Monitor initialized")
    
    def _default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Default performance thresholds"""
        return {
            "response_time": {
                "warning": 3.0,    # seconds
                "error": 5.0,      # seconds
                "critical": 10.0   # seconds
            },
            "success_rate": {
                "warning": 95.0,   # percentage
                "error": 90.0,     # percentage
                "critical": 80.0   # percentage
            },
            "memory_usage": {
                "warning": 70.0,   # percentage
                "error": 85.0,     # percentage
                "critical": 95.0   # percentage
            },
            "cpu_usage": {
                "warning": 70.0,   # percentage
                "error": 85.0,     # percentage
                "critical": 95.0   # percentage
            },
            "disk_usage": {
                "warning": 80.0,   # percentage
                "error": 90.0,     # percentage
                "critical": 95.0   # percentage
            },
            "error_rate": {
                "warning": 5.0,    # percentage
                "error": 10.0,     # percentage
                "critical": 20.0   # percentage
            }
        }
    
    def set_components(self, query_repository, vector_store, cache):
        """Set component references for monitoring"""
        self.query_repository = query_repository
        self.vector_store = vector_store
        self.cache = cache
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback function for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return
        
        if not self.monitoring_enabled:
            logger.info("Performance monitoring disabled in configuration")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                self._store_metrics(metrics)
                self._check_thresholds(metrics)
                self.last_metrics = metrics
                
                # Record system snapshot if repository is available
                if self.query_repository:
                    asyncio.run(self._record_system_snapshot(metrics))
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics"""
        try:
            # System resource usage
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')
            
            # Calculate performance metrics from recent data
            response_time = self._calculate_avg_response_time()
            success_rate = self._calculate_success_rate()
            error_rate = 100.0 - success_rate
            cache_hit_rate = self._calculate_cache_hit_rate()
            active_users = self._count_active_users()
            query_volume = self._count_recent_queries()
            
            return SystemMetrics(
                timestamp=datetime.utcnow(),
                response_time=response_time,
                success_rate=success_rate,
                error_rate=error_rate,
                memory_usage=memory.percent,
                cpu_usage=cpu_percent,
                disk_usage=disk.percent,
                cache_hit_rate=cache_hit_rate,
                active_users=active_users,
                query_volume=query_volume,
                system_load=os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.utcnow(),
                response_time=0.0, success_rate=0.0, error_rate=100.0,
                memory_usage=0.0, cpu_usage=0.0, disk_usage=0.0,
                cache_hit_rate=0.0, active_users=0, query_volume=0,
                system_load=0.0
            )
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time from recent queries"""
        try:
            if not self.query_repository:
                return 0.0
            
            # This would need to be made async in a real implementation
            # For now, return a mock value
            return 1.2  # Mock average response time
            
        except Exception as e:
            logger.error(f"Error calculating response time: {e}")
            return 0.0
    
    def _calculate_success_rate(self) -> float:
        """Calculate success rate from recent queries"""
        try:
            if not self.query_repository:
                return 100.0
            
            # Mock success rate - in real implementation, would query recent data
            return 98.5
            
        except Exception as e:
            logger.error(f"Error calculating success rate: {e}")
            return 0.0
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        try:
            if not self.cache:
                return 0.0
            
            # Get cache statistics if available
            cache_stats = getattr(self.cache, 'get_stats', lambda: {})()
            
            total_requests = cache_stats.get('total_requests', 0)
            cache_hits = cache_stats.get('cache_hits', 0)
            
            if total_requests > 0:
                return (cache_hits / total_requests) * 100
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating cache hit rate: {e}")
            return 0.0
    
    def _count_active_users(self) -> int:
        """Count active users in recent time window"""
        try:
            if not self.query_repository:
                return 0
            
            # Mock active user count
            return 15
            
        except Exception as e:
            logger.error(f"Error counting active users: {e}")
            return 0
    
    def _count_recent_queries(self) -> int:
        """Count queries in recent time window"""
        try:
            if not self.query_repository:
                return 0
            
            # Mock query volume
            return 42
            
        except Exception as e:
            logger.error(f"Error counting recent queries: {e}")
            return 0
    
    def _store_metrics(self, metrics: SystemMetrics):
        """Store metrics in memory history"""
        self.metrics_history.append(metrics)
        
        # Trim history if too large
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    def _check_thresholds(self, metrics: SystemMetrics):
        """Check metrics against thresholds and generate alerts"""
        checks = [
            (MetricType.RESPONSE_TIME, metrics.response_time, "response_time"),
            (MetricType.SUCCESS_RATE, metrics.success_rate, "success_rate"),
            (MetricType.MEMORY_USAGE, metrics.memory_usage, "memory_usage"),
            (MetricType.CPU_USAGE, metrics.cpu_usage, "cpu_usage"),
            (MetricType.DISK_USAGE, metrics.disk_usage, "disk_usage"),
            (MetricType.ERROR_RATE, metrics.error_rate, "error_rate")
        ]
        
        for metric_type, value, threshold_key in checks:
            alert = self._check_metric_threshold(metric_type, value, threshold_key)
            if alert:
                self._trigger_alert(alert)
    
    def _check_metric_threshold(
        self, 
        metric_type: MetricType, 
        value: float, 
        threshold_key: str
    ) -> Optional[PerformanceAlert]:
        """Check if a metric exceeds its thresholds"""
        thresholds = self.alert_thresholds.get(threshold_key, {})
        
        # Determine alert level
        alert_level = None
        threshold_value = None
        
        if metric_type == MetricType.SUCCESS_RATE:
            # For success rate, lower values are worse
            if value <= thresholds.get("critical", 0):
                alert_level = AlertLevel.CRITICAL
                threshold_value = thresholds["critical"]
            elif value <= thresholds.get("error", 0):
                alert_level = AlertLevel.ERROR
                threshold_value = thresholds["error"]
            elif value <= thresholds.get("warning", 0):
                alert_level = AlertLevel.WARNING
                threshold_value = thresholds["warning"]
        else:
            # For other metrics, higher values are worse
            if value >= thresholds.get("critical", float('inf')):
                alert_level = AlertLevel.CRITICAL
                threshold_value = thresholds["critical"]
            elif value >= thresholds.get("error", float('inf')):
                alert_level = AlertLevel.ERROR
                threshold_value = thresholds["error"]
            elif value >= thresholds.get("warning", float('inf')):
                alert_level = AlertLevel.WARNING
                threshold_value = thresholds["warning"]
        
        if alert_level:
            return PerformanceAlert(
                level=alert_level,
                metric=metric_type,
                current_value=value,
                threshold=threshold_value,
                message=self._generate_alert_message(metric_type, value, threshold_value),
                timestamp=datetime.utcnow(),
                affected_components=self._get_affected_components(metric_type),
                suggested_actions=self._get_suggested_actions(metric_type, alert_level)
            )
        
        return None
    
    def _generate_alert_message(
        self, 
        metric_type: MetricType, 
        value: float, 
        threshold: float
    ) -> str:
        """Generate human-readable alert message"""
        metric_name = metric_type.value.replace('_', ' ').title()
        
        if metric_type == MetricType.SUCCESS_RATE:
            return f"{metric_name} has dropped to {value:.1f}% (threshold: {threshold:.1f}%)"
        else:
            return f"{metric_name} has reached {value:.1f} (threshold: {threshold:.1f})"
    
    def _get_affected_components(self, metric_type: MetricType) -> List[str]:
        """Get components likely affected by this metric"""
        component_map = {
            MetricType.RESPONSE_TIME: ["orchestrator", "agents", "vector_store"],
            MetricType.SUCCESS_RATE: ["all_components"],
            MetricType.ERROR_RATE: ["all_components"],
            MetricType.MEMORY_USAGE: ["vector_store", "cache", "database"],
            MetricType.CPU_USAGE: ["orchestrator", "agents"],
            MetricType.DISK_USAGE: ["database", "vector_store", "cache"],
            MetricType.CACHE_HIT_RATE: ["cache"]
        }
        
        return component_map.get(metric_type, ["unknown"])
    
    def _get_suggested_actions(
        self, 
        metric_type: MetricType, 
        alert_level: AlertLevel
    ) -> List[str]:
        """Get suggested actions for metric alerts"""
        action_map = {
            MetricType.RESPONSE_TIME: [
                "Check network connectivity",
                "Optimize query complexity",
                "Review agent performance",
                "Consider caching improvements"
            ],
            MetricType.SUCCESS_RATE: [
                "Check error logs",
                "Verify API key validity",
                "Review recent system changes",
                "Monitor external service status"
            ],
            MetricType.MEMORY_USAGE: [
                "Clear unused cache entries",
                "Optimize vector store",
                "Check for memory leaks",
                "Consider increasing system memory"
            ],
            MetricType.CPU_USAGE: [
                "Reduce concurrent operations",
                "Optimize agent algorithms",
                "Check for infinite loops",
                "Consider horizontal scaling"
            ],
            MetricType.DISK_USAGE: [
                "Clean up old log files",
                "Archive old data",
                "Optimize database storage",
                "Add disk space"
            ]
        }
        
        base_actions = action_map.get(metric_type, ["Monitor situation"])
        
        if alert_level == AlertLevel.CRITICAL:
            base_actions.insert(0, "Immediate action required")
        
        return base_actions
    
    def _trigger_alert(self, alert: PerformanceAlert):
        """Trigger alert notifications"""
        self.alerts_history.append(alert)
        
        # Trim alert history
        if len(self.alerts_history) > 100:
            self.alerts_history = self.alerts_history[-100:]
        
        # Log alert
        logger.log(
            logging.ERROR if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL] else logging.WARNING,
            f"Performance Alert [{alert.level.value.upper()}]: {alert.message}"
        )
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    async def _record_system_snapshot(self, metrics: SystemMetrics):
        """Record system snapshot in query repository"""
        try:
            snapshot_data = {
                "memory_usage": metrics.memory_usage,
                "cpu_usage": metrics.cpu_usage,
                "disk_usage": metrics.disk_usage,
                "active_connections": metrics.active_users,
                "cache_size": 0,  # Would need actual cache size
                "database_size": 0,  # Would need actual DB size
                "vector_store_size": 0,  # Would need actual vector store size
                "system_status": self._determine_system_status(metrics),
                "notes": f"Query volume: {metrics.query_volume}, Load: {metrics.system_load}"
            }
            
            await self.query_repository.record_system_snapshot(snapshot_data)
            
        except Exception as e:
            logger.error(f"Error recording system snapshot: {e}")
    
    def _determine_system_status(self, metrics: SystemMetrics) -> str:
        """Determine overall system status"""
        if (metrics.memory_usage > 90 or 
            metrics.cpu_usage > 90 or 
            metrics.disk_usage > 95 or
            metrics.error_rate > 20):
            return "critical"
        elif (metrics.memory_usage > 80 or 
              metrics.cpu_usage > 80 or 
              metrics.disk_usage > 90 or
              metrics.error_rate > 10):
            return "warning"
        elif metrics.success_rate > 95:
            return "healthy"
        else:
            return "degraded"
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status"""
        if not self.last_metrics:
            return {"status": "unknown", "message": "No metrics available"}
        
        metrics = self.last_metrics
        status = self._determine_system_status(metrics)
        
        return {
            "status": status,
            "timestamp": metrics.timestamp.isoformat(),
            "metrics": {
                "response_time": metrics.response_time,
                "success_rate": metrics.success_rate,
                "memory_usage": metrics.memory_usage,
                "cpu_usage": metrics.cpu_usage,
                "disk_usage": metrics.disk_usage,
                "cache_hit_rate": metrics.cache_hit_rate,
                "active_users": metrics.active_users,
                "query_volume": metrics.query_volume
            },
            "recent_alerts": len([a for a in self.alerts_history 
                                if (datetime.utcnow() - a.timestamp).seconds < 3600])
        }
    
    def get_performance_trends(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get performance trends over time"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {"status": "no_data", "message": "No metrics available for time period"}
        
        # Calculate trends
        response_times = [m.response_time for m in recent_metrics]
        success_rates = [m.success_rate for m in recent_metrics]
        memory_usage = [m.memory_usage for m in recent_metrics]
        
        return {
            "period_hours": hours_back,
            "total_samples": len(recent_metrics),
            "trends": {
                "response_time": {
                    "avg": sum(response_times) / len(response_times),
                    "min": min(response_times),
                    "max": max(response_times),
                    "trend": "stable"  # Would calculate actual trend
                },
                "success_rate": {
                    "avg": sum(success_rates) / len(success_rates),
                    "min": min(success_rates),
                    "max": max(success_rates),
                    "trend": "stable"
                },
                "memory_usage": {
                    "avg": sum(memory_usage) / len(memory_usage),
                    "min": min(memory_usage),
                    "max": max(memory_usage),
                    "trend": "stable"
                }
            },
            "alerts": {
                "total": len(self.alerts_history),
                "recent": len([a for a in self.alerts_history 
                             if (datetime.utcnow() - a.timestamp).seconds < hours_back * 3600])
            }
        }
    
    def get_alerts_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get summary of recent alerts"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        recent_alerts = [a for a in self.alerts_history if a.timestamp > cutoff_time]
        
        if not recent_alerts:
            return {"total": 0, "by_level": {}, "by_metric": {}}
        
        # Group by level and metric
        by_level = {}
        by_metric = {}
        
        for alert in recent_alerts:
            level = alert.level.value
            metric = alert.metric.value
            
            by_level[level] = by_level.get(level, 0) + 1
            by_metric[metric] = by_metric.get(metric, 0) + 1
        
        return {
            "total": len(recent_alerts),
            "by_level": by_level,
            "by_metric": by_metric,
            "most_recent": {
                "level": recent_alerts[-1].level.value,
                "metric": recent_alerts[-1].metric.value,
                "message": recent_alerts[-1].message,
                "timestamp": recent_alerts[-1].timestamp.isoformat()
            } if recent_alerts else None
        }
