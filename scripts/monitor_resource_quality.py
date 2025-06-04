#!/usr/bin/env python3
"""
Real-time Resource Detection Quality Monitor

Continuous monitoring system for resource detection and classification quality.
Provides ongoing evaluation of the system's performance with real Discord data.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import sqlite3
from collections import defaultdict, deque

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.services.content_processor import ContentProcessingService
from agentic.services.unified_data_manager import UnifiedDataManager

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Quality metrics for monitoring"""
    timestamp: str
    total_messages_processed: int
    urls_detected: int
    code_snippets_detected: int
    attachments_processed: int
    ai_classifications_successful: int
    ai_classifications_failed: int
    processing_errors: int
    average_processing_time: float
    quality_score: float

@dataclass
class ResourceTypeDistribution:
    """Distribution of detected resource types"""
    documents: int = 0
    code: int = 0
    development: int = 0
    video: int = 0
    unknown: int = 0
    educational: int = 0
    research: int = 0

class ResourceQualityMonitor:
    """
    Real-time quality monitoring for resource detection system
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.data_manager = UnifiedDataManager(self.config.get('data_manager', {}))
        
        # Initialize monitoring database
        self.monitor_db_path = project_root / "data" / "resource_quality_monitor.db"
        self._init_monitoring_db()
        
        # Metrics storage
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 metrics
        self.current_session_stats = {
            "start_time": datetime.now(),
            "messages_processed": 0,
            "total_processing_time": 0.0,
            "errors": [],
            "quality_issues": []
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            "min_processing_success_rate": 0.95,
            "max_average_processing_time": 2.0,  # seconds
            "min_classification_success_rate": 0.90,
            "min_overall_quality_score": 0.75
        }
        
        logger.info("🔍 Resource Quality Monitor initialized")
    
    def _init_monitoring_db(self):
        """Initialize monitoring database"""
        self.monitor_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.monitor_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_messages_processed INTEGER,
                    urls_detected INTEGER,
                    code_snippets_detected INTEGER,
                    attachments_processed INTEGER,
                    ai_classifications_successful INTEGER,
                    ai_classifications_failed INTEGER,
                    processing_errors INTEGER,
                    average_processing_time REAL,
                    quality_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS resource_type_distribution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    documents INTEGER DEFAULT 0,
                    code INTEGER DEFAULT 0,
                    development INTEGER DEFAULT 0,
                    video INTEGER DEFAULT 0,
                    unknown INTEGER DEFAULT 0,
                    educational INTEGER DEFAULT 0,
                    research INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metrics_snapshot TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    async def monitor_resource_detection_quality(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Monitor resource detection quality over a specified time window
        """
        print(f"🔍 Monitoring resource detection quality (last {time_window_hours} hours)")
        
        # Get recent messages
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        messages = await self._get_recent_messages(cutoff_time)
        
        if not messages:
            print("⚠️ No recent messages found for monitoring")
            return {"error": "No recent messages available"}
        
        # Initialize content processor for testing
        try:
            from openai import OpenAI
            openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            content_processor = ContentProcessingService(openai_client)
        except Exception as e:
            print(f"❌ Failed to initialize content processor: {e}")
            return {"error": f"Content processor initialization failed: {e}"}
        
        # Process messages and collect metrics
        metrics = await self._process_messages_for_quality(messages, content_processor)
        
        # Analyze quality and generate report
        quality_report = self._analyze_quality_metrics(metrics)
        
        # Store metrics in database
        await self._store_quality_metrics(metrics)
        
        # Check for quality alerts
        alerts = self._check_quality_alerts(metrics)
        if alerts:
            await self._store_quality_alerts(alerts)
        
        return {
            "monitoring_period": f"{time_window_hours} hours",
            "messages_analyzed": len(messages),
            "metrics": asdict(metrics),
            "quality_report": quality_report,
            "alerts": alerts,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_recent_messages(self, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Get recent messages from data manager"""
        try:
            await self.data_manager.initialize()
            
            # Query recent messages from SQLite
            query = """
                SELECT message_id, content, channel_name, author_username, timestamp, attachments
                FROM messages 
                WHERE datetime(timestamp) > ?
                ORDER BY timestamp DESC
                LIMIT 500
            """
            
            messages = await self.data_manager.query_data(
                'sqlite', 
                query, 
                (cutoff_time.isoformat(),)
            )
            
            return messages or []
            
        except Exception as e:
            logger.error(f"Error getting recent messages: {e}")
            return []
    
    async def _process_messages_for_quality(self, messages: List[Dict[str, Any]], content_processor: ContentProcessingService) -> QualityMetrics:
        """Process messages and collect quality metrics"""
        start_time = datetime.now()
        
        total_processed = 0
        urls_detected = 0
        code_snippets_detected = 0
        attachments_processed = 0
        ai_success = 0
        ai_failed = 0
        processing_errors = 0
        processing_times = []
        
        resource_distribution = ResourceTypeDistribution()
        
        print(f"📊 Processing {len(messages)} messages for quality analysis...")
        
        for i, message in enumerate(messages):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(messages)} messages processed")
            
            try:
                msg_start = datetime.now()
                
                # Prepare message data
                message_data = {
                    "message_id": message.get("message_id"),
                    "content": message.get("content", ""),
                    "attachments": json.loads(message.get("attachments", "[]")),
                    "channel_name": message.get("channel_name"),
                    "author": {"username": message.get("author_username")},
                    "timestamp": message.get("timestamp")
                }
                
                # Skip empty messages
                if not message_data["content"] and not message_data["attachments"]:
                    continue
                
                # Process with content processor
                analysis = await content_processor.analyze_message_content(message_data)
                
                # Collect metrics
                total_processed += 1
                urls_detected += len(analysis.get("urls", []))
                code_snippets_detected += len(analysis.get("code_snippets", []))
                attachments_processed += len(analysis.get("attachments_processed", []))
                
                # Track AI classification success/failure
                classifications = analysis.get("classifications", [])
                if classifications and "unclassified" not in classifications:
                    ai_success += 1
                elif classifications and "unclassified" in classifications:
                    ai_failed += 1
                
                # Track resource type distribution
                for url_data in analysis.get("urls", []):
                    resource_type = url_data.get("type", "unknown")
                    if hasattr(resource_distribution, resource_type):
                        setattr(resource_distribution, resource_type, 
                               getattr(resource_distribution, resource_type) + 1)
                    else:
                        resource_distribution.unknown += 1
                
                # Track processing time
                processing_time = (datetime.now() - msg_start).total_seconds()
                processing_times.append(processing_time)
                
            except Exception as e:
                processing_errors += 1
                logger.debug(f"Error processing message {message.get('message_id')}: {e}")
        
        # Calculate metrics
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        
        # Calculate quality score (simplified heuristic)
        success_rate = (total_processed - processing_errors) / total_processed if total_processed > 0 else 0.0
        ai_success_rate = ai_success / (ai_success + ai_failed) if (ai_success + ai_failed) > 0 else 1.0
        resource_density = (urls_detected + code_snippets_detected + attachments_processed) / total_processed if total_processed > 0 else 0.0
        
        quality_score = (success_rate * 0.4 + ai_success_rate * 0.3 + min(1.0, resource_density) * 0.3)
        
        end_time = datetime.now()
        
        print(f"✅ Quality analysis complete in {(end_time - start_time).total_seconds():.2f} seconds")
        
        return QualityMetrics(
            timestamp=datetime.now().isoformat(),
            total_messages_processed=total_processed,
            urls_detected=urls_detected,
            code_snippets_detected=code_snippets_detected,
            attachments_processed=attachments_processed,
            ai_classifications_successful=ai_success,
            ai_classifications_failed=ai_failed,
            processing_errors=processing_errors,
            average_processing_time=avg_processing_time,
            quality_score=quality_score
        )
    
    def _analyze_quality_metrics(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """Analyze quality metrics and generate insights"""
        total_resources = metrics.urls_detected + metrics.code_snippets_detected + metrics.attachments_processed
        total_ai_attempts = metrics.ai_classifications_successful + metrics.ai_classifications_failed
        
        analysis = {
            "overall_quality": self._get_quality_rating(metrics.quality_score),
            "processing_performance": {
                "success_rate": (metrics.total_messages_processed - metrics.processing_errors) / metrics.total_messages_processed if metrics.total_messages_processed > 0 else 0.0,
                "average_processing_time": metrics.average_processing_time,
                "performance_rating": "Good" if metrics.average_processing_time < 1.0 else "Needs Improvement"
            },
            "resource_detection": {
                "total_resources_found": total_resources,
                "resources_per_message": total_resources / metrics.total_messages_processed if metrics.total_messages_processed > 0 else 0.0,
                "url_detection_rate": metrics.urls_detected / metrics.total_messages_processed if metrics.total_messages_processed > 0 else 0.0
            },
            "ai_classification": {
                "success_rate": metrics.ai_classifications_successful / total_ai_attempts if total_ai_attempts > 0 else 0.0,
                "total_attempts": total_ai_attempts,
                "classification_rating": "Good" if total_ai_attempts > 0 and metrics.ai_classifications_successful / total_ai_attempts > 0.9 else "Needs Improvement"
            },
            "recommendations": self._generate_quality_recommendations(metrics)
        }
        
        return analysis
    
    def _get_quality_rating(self, score: float) -> str:
        """Convert quality score to rating"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Good"
        elif score >= 0.7:
            return "Fair"
        elif score >= 0.6:
            return "Poor"
        else:
            return "Critical"
    
    def _generate_quality_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        if metrics.processing_errors > metrics.total_messages_processed * 0.05:
            recommendations.append("High error rate detected - investigate processing pipeline stability")
        
        if metrics.average_processing_time > 2.0:
            recommendations.append("Processing time is above target - consider performance optimization")
        
        total_ai_attempts = metrics.ai_classifications_successful + metrics.ai_classifications_failed
        if total_ai_attempts > 0 and metrics.ai_classifications_failed / total_ai_attempts > 0.1:
            recommendations.append("AI classification failure rate is high - check API connectivity and prompts")
        
        if metrics.quality_score < 0.75:
            recommendations.append("Overall quality score is below target - review detection algorithms")
        
        total_resources = metrics.urls_detected + metrics.code_snippets_detected + metrics.attachments_processed
        if total_resources / metrics.total_messages_processed < 0.1 if metrics.total_messages_processed > 0 else True:
            recommendations.append("Resource detection rate is low - consider expanding detection patterns")
        
        if not recommendations:
            recommendations.append("System performance is within acceptable parameters")
        
        return recommendations
    
    def _check_quality_alerts(self, metrics: QualityMetrics) -> List[Dict[str, Any]]:
        """Check for quality alerts based on thresholds"""
        alerts = []
        
        # Processing success rate alert
        success_rate = (metrics.total_messages_processed - metrics.processing_errors) / metrics.total_messages_processed if metrics.total_messages_processed > 0 else 0.0
        if success_rate < self.quality_thresholds["min_processing_success_rate"]:
            alerts.append({
                "type": "processing_success_rate",
                "severity": "high",
                "message": f"Processing success rate ({success_rate:.2%}) below threshold ({self.quality_thresholds['min_processing_success_rate']:.2%})",
                "value": success_rate,
                "threshold": self.quality_thresholds["min_processing_success_rate"]
            })
        
        # Processing time alert
        if metrics.average_processing_time > self.quality_thresholds["max_average_processing_time"]:
            alerts.append({
                "type": "processing_time",
                "severity": "medium",
                "message": f"Average processing time ({metrics.average_processing_time:.2f}s) above threshold ({self.quality_thresholds['max_average_processing_time']}s)",
                "value": metrics.average_processing_time,
                "threshold": self.quality_thresholds["max_average_processing_time"]
            })
        
        # AI classification success rate alert
        total_ai_attempts = metrics.ai_classifications_successful + metrics.ai_classifications_failed
        if total_ai_attempts > 0:
            ai_success_rate = metrics.ai_classifications_successful / total_ai_attempts
            if ai_success_rate < self.quality_thresholds["min_classification_success_rate"]:
                alerts.append({
                    "type": "ai_classification_success_rate",
                    "severity": "high",
                    "message": f"AI classification success rate ({ai_success_rate:.2%}) below threshold ({self.quality_thresholds['min_classification_success_rate']:.2%})",
                    "value": ai_success_rate,
                    "threshold": self.quality_thresholds["min_classification_success_rate"]
                })
        
        # Overall quality score alert
        if metrics.quality_score < self.quality_thresholds["min_overall_quality_score"]:
            alerts.append({
                "type": "overall_quality_score",
                "severity": "high",
                "message": f"Overall quality score ({metrics.quality_score:.2f}) below threshold ({self.quality_thresholds['min_overall_quality_score']})",
                "value": metrics.quality_score,
                "threshold": self.quality_thresholds["min_overall_quality_score"]
            })
        
        return alerts
    
    async def _store_quality_metrics(self, metrics: QualityMetrics):
        """Store quality metrics in database"""
        try:
            with sqlite3.connect(self.monitor_db_path) as conn:
                conn.execute("""
                    INSERT INTO quality_metrics (
                        timestamp, total_messages_processed, urls_detected, 
                        code_snippets_detected, attachments_processed,
                        ai_classifications_successful, ai_classifications_failed,
                        processing_errors, average_processing_time, quality_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp, metrics.total_messages_processed,
                    metrics.urls_detected, metrics.code_snippets_detected,
                    metrics.attachments_processed, metrics.ai_classifications_successful,
                    metrics.ai_classifications_failed, metrics.processing_errors,
                    metrics.average_processing_time, metrics.quality_score
                ))
        except Exception as e:
            logger.error(f"Error storing quality metrics: {e}")
    
    async def _store_quality_alerts(self, alerts: List[Dict[str, Any]]):
        """Store quality alerts in database"""
        try:
            with sqlite3.connect(self.monitor_db_path) as conn:
                for alert in alerts:
                    conn.execute("""
                        INSERT INTO quality_alerts (
                            timestamp, alert_type, severity, message, metrics_snapshot
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        datetime.now().isoformat(),
                        alert["type"],
                        alert["severity"],
                        alert["message"],
                        json.dumps(alert)
                    ))
        except Exception as e:
            logger.error(f"Error storing quality alerts: {e}")
    
    async def get_quality_history(self, days: int = 7) -> Dict[str, Any]:
        """Get quality metrics history"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.monitor_db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute("""
                    SELECT * FROM quality_metrics 
                    WHERE datetime(timestamp) > ?
                    ORDER BY timestamp DESC
                """, (cutoff_time.isoformat(),))
                
                metrics_history = [dict(row) for row in cursor.fetchall()]
                
                cursor = conn.execute("""
                    SELECT * FROM quality_alerts 
                    WHERE datetime(timestamp) > ? AND resolved = FALSE
                    ORDER BY timestamp DESC
                """, (cutoff_time.isoformat(),))
                
                active_alerts = [dict(row) for row in cursor.fetchall()]
            
            return {
                "period_days": days,
                "metrics_count": len(metrics_history),
                "metrics_history": metrics_history,
                "active_alerts": active_alerts,
                "summary": self._generate_history_summary(metrics_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting quality history: {e}")
            return {"error": str(e)}
    
    def _generate_history_summary(self, metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary from metrics history"""
        if not metrics_history:
            return {"message": "No metrics available"}
        
        quality_scores = [m["quality_score"] for m in metrics_history if m["quality_score"]]
        processing_times = [m["average_processing_time"] for m in metrics_history if m["average_processing_time"]]
        
        return {
            "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            "quality_trend": "improving" if len(quality_scores) > 1 and quality_scores[0] > quality_scores[-1] else "declining" if len(quality_scores) > 1 else "stable",
            "average_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0.0,
            "total_messages_processed": sum(m["total_messages_processed"] for m in metrics_history),
            "total_resources_detected": sum(m["urls_detected"] + m["code_snippets_detected"] + m["attachments_processed"] for m in metrics_history)
        }

# Main execution functions
async def run_quality_monitoring():
    """Run quality monitoring"""
    monitor = ResourceQualityMonitor()
    
    print("🚀 Starting Resource Detection Quality Monitoring")
    print("=" * 60)
    
    # Run 24-hour monitoring
    results = await monitor.monitor_resource_detection_quality(time_window_hours=24)
    
    if "error" in results:
        print(f"❌ Monitoring failed: {results['error']}")
        return
    
    # Display results
    print(f"\n📊 QUALITY MONITORING RESULTS")
    print(f"Period: {results['monitoring_period']}")
    print(f"Messages Analyzed: {results['messages_analyzed']}")
    
    metrics = results['metrics']
    print(f"\n📈 KEY METRICS:")
    print(f"  Quality Score: {metrics['quality_score']:.3f}")
    print(f"  Messages Processed: {metrics['total_messages_processed']}")
    print(f"  URLs Detected: {metrics['urls_detected']}")
    print(f"  Code Snippets: {metrics['code_snippets_detected']}")
    print(f"  Attachments: {metrics['attachments_processed']}")
    print(f"  AI Success Rate: {metrics['ai_classifications_successful']}/{metrics['ai_classifications_successful'] + metrics['ai_classifications_failed']}")
    print(f"  Processing Errors: {metrics['processing_errors']}")
    print(f"  Avg Processing Time: {metrics['average_processing_time']:.3f}s")
    
    quality_report = results['quality_report']
    print(f"\n🎯 QUALITY ANALYSIS:")
    print(f"  Overall Quality: {quality_report['overall_quality']}")
    print(f"  Processing Performance: {quality_report['processing_performance']['performance_rating']}")
    print(f"  AI Classification: {quality_report['ai_classification']['classification_rating']}")
    
    print(f"\n🔧 RECOMMENDATIONS:")
    for rec in quality_report['recommendations']:
        print(f"  💡 {rec}")
    
    # Display alerts if any
    if results['alerts']:
        print(f"\n🚨 QUALITY ALERTS:")
        for alert in results['alerts']:
            print(f"  {alert['severity'].upper()}: {alert['message']}")
    else:
        print(f"\n✅ No quality alerts - system performing within parameters")
    
    # Save results
    output_file = f"quality_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = project_root / "data" / "exports" / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")

async def get_quality_dashboard():
    """Get quality dashboard data"""
    monitor = ResourceQualityMonitor()
    history = await monitor.get_quality_history(days=7)
    
    print("📊 QUALITY DASHBOARD (Last 7 Days)")
    print("=" * 50)
    
    if "error" in history:
        print(f"❌ Error: {history['error']}")
        return
    
    summary = history['summary']
    print(f"Average Quality Score: {summary['average_quality_score']:.3f}")
    print(f"Quality Trend: {summary['quality_trend'].title()}")
    print(f"Average Processing Time: {summary['average_processing_time']:.3f}s")
    print(f"Total Messages Processed: {summary['total_messages_processed']}")
    print(f"Total Resources Detected: {summary['total_resources_detected']}")
    
    if history['active_alerts']:
        print(f"\n🚨 ACTIVE ALERTS ({len(history['active_alerts'])}):")
        for alert in history['active_alerts'][:5]:  # Show latest 5
            print(f"  {alert['severity'].upper()}: {alert['message']}")
    else:
        print(f"\n✅ No active alerts")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Resource Detection Quality Monitor")
    parser.add_argument("--dashboard", action="store_true", help="Show quality dashboard")
    parser.add_argument("--monitor", action="store_true", help="Run quality monitoring")
    
    args = parser.parse_args()
    
    if args.dashboard:
        asyncio.run(get_quality_dashboard())
    elif args.monitor:
        asyncio.run(run_quality_monitoring())
    else:
        # Default: run monitoring
        asyncio.run(run_quality_monitoring())
