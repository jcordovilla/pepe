#!/usr/bin/env python3
"""
Enhanced Resource Detection Quality Monitor

Real-time monitoring system for resource detection and classification quality.
Provides continuous assessment, alerting, and improvement recommendations.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time
from dataclasses import dataclass
import statistics

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.services.content_processor import ContentProcessingService
from openai import OpenAI

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Quality metrics for monitoring"""
    timestamp: datetime
    url_detection_rate: float
    code_detection_rate: float
    attachment_processing_rate: float
    ai_classification_success_rate: float
    noise_filtering_accuracy: float
    overall_quality_score: float
    
@dataclass
class QualityAlert:
    """Quality alert for degradation detection"""
    timestamp: datetime
    component: str
    severity: str  # "low", "medium", "high", "critical"
    metric: str
    current_value: float
    threshold: float
    description: str
    recommendation: str

class ResourceQualityMonitor:
    """
    Real-time resource detection quality monitor
    
    Features:
    - Continuous quality assessment
    - Performance degradation detection  
    - Automated alerting
    - Quality trend analysis
    - Improvement recommendations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.content_processor = ContentProcessingService(self.openai_client)
        
        # Quality thresholds for alerting
        self.thresholds = self.config.get('thresholds', {
            'url_detection_rate': 0.8,
            'code_detection_rate': 0.7,
            'attachment_processing_rate': 0.95,
            'ai_classification_success_rate': 0.6,
            'noise_filtering_accuracy': 0.9,
            'overall_quality_score': 0.75
        })
        
        # Data storage
        self.metrics_history: List[QualityMetrics] = []
        self.alerts: List[QualityAlert] = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_interval = self.config.get('monitoring_interval', 300)  # 5 minutes
        
        # Load existing data if available
        self._load_historical_data()
        
        logger.info("🔍 Resource Quality Monitor initialized")
    
    async def start_monitoring(self, duration_minutes: Optional[int] = None):
        """Start continuous quality monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
        
        self.is_monitoring = True
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes) if duration_minutes else None
        
        print(f"🚀 Starting resource quality monitoring...")
        if duration_minutes:
            print(f"⏰ Monitoring duration: {duration_minutes} minutes")
        else:
            print("⏰ Continuous monitoring (Ctrl+C to stop)")
        
        try:
            while self.is_monitoring:
                if end_time and datetime.now() >= end_time:
                    break
                
                # Collect current metrics
                metrics = await self._collect_current_metrics()
                self.metrics_history.append(metrics)
                
                # Check for quality degradation
                alerts = self._check_quality_thresholds(metrics)
                self.alerts.extend(alerts)
                
                # Display current status
                self._display_current_status(metrics, alerts)
                
                # Save data periodically
                if len(self.metrics_history) % 5 == 0:
                    self._save_monitoring_data()
                
                # Wait for next check
                await asyncio.sleep(self.monitoring_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
        finally:
            self.is_monitoring = False
            self._save_monitoring_data()
            await self._generate_monitoring_report()
    
    async def _collect_current_metrics(self) -> QualityMetrics:
        """Collect current quality metrics from recent data"""
        
        # Sample recent messages for quality assessment
        sample_messages = await self._get_recent_sample_messages()
        
        if not sample_messages:
            # Return baseline metrics if no recent data
            return QualityMetrics(
                timestamp=datetime.now(),
                url_detection_rate=0.0,
                code_detection_rate=0.0,
                attachment_processing_rate=0.0,
                ai_classification_success_rate=0.0,
                noise_filtering_accuracy=0.0,
                overall_quality_score=0.0
            )
        
        # Process sample messages
        url_detections = []
        code_detections = []
        attachment_processings = []
        ai_classifications = []
        noise_filterings = []
        
        for message in sample_messages:
            try:
                analysis = await self.content_processor.analyze_message_content(message)
                
                # URL detection assessment
                content = message.get('content', '')
                has_urls = 'http' in content
                detected_urls = len(analysis.get('urls', [])) > 0
                url_detections.append(1.0 if (has_urls == detected_urls) else 0.0)
                
                # Code detection assessment
                has_code = '```' in content or '`' in content
                detected_code = len(analysis.get('code_snippets', [])) > 0
                code_detections.append(1.0 if (has_code == detected_code) else 0.0)
                
                # Attachment processing assessment
                has_attachments = len(message.get('attachments', [])) > 0
                processed_attachments = len(analysis.get('attachments_processed', [])) > 0
                attachment_processings.append(1.0 if (has_attachments == processed_attachments) else 0.0)
                
                # AI classification assessment (success = non-empty result)
                classifications = analysis.get('classifications', [])
                ai_classifications.append(1.0 if classifications and classifications != ['unclassified'] else 0.0)
                
                # Noise filtering assessment (check if noise domains are filtered)
                urls = analysis.get('urls', [])
                noise_domains = {'tenor.com', 'giphy.com', 'discord.com', 'cdn.discordapp.com'}
                has_noise = any(domain in content.lower() for domain in noise_domains)
                filtered_noise = not any(url.get('domain', '') in noise_domains for url in urls)
                noise_filterings.append(1.0 if (not has_noise or filtered_noise) else 0.0)
                
            except Exception as e:
                logger.debug(f"Error processing message for metrics: {e}")
                # Add 0 scores for failed processing
                url_detections.append(0.0)
                code_detections.append(0.0)
                attachment_processings.append(0.0)
                ai_classifications.append(0.0)
                noise_filterings.append(0.0)
        
        # Calculate metrics
        url_detection_rate = statistics.mean(url_detections) if url_detections else 0.0
        code_detection_rate = statistics.mean(code_detections) if code_detections else 0.0
        attachment_processing_rate = statistics.mean(attachment_processings) if attachment_processings else 0.0
        ai_classification_success_rate = statistics.mean(ai_classifications) if ai_classifications else 0.0
        noise_filtering_accuracy = statistics.mean(noise_filterings) if noise_filterings else 0.0
        
        overall_quality_score = statistics.mean([
            url_detection_rate,
            code_detection_rate,
            attachment_processing_rate,
            ai_classification_success_rate,
            noise_filtering_accuracy
        ])
        
        return QualityMetrics(
            timestamp=datetime.now(),
            url_detection_rate=url_detection_rate,
            code_detection_rate=code_detection_rate,
            attachment_processing_rate=attachment_processing_rate,
            ai_classification_success_rate=ai_classification_success_rate,
            noise_filtering_accuracy=noise_filtering_accuracy,
            overall_quality_score=overall_quality_score
        )
    
    async def _get_recent_sample_messages(self, sample_size: int = 20) -> List[Dict[str, Any]]:
        """Get recent messages for quality assessment"""
        try:
            # Load recent messages from the Discord message files
            messages_dir = project_root / "data" / "fetched_messages"
            if not messages_dir.exists():
                return []
            
            # Get most recent message files
            message_files = sorted(messages_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
            
            sample_messages = []
            for file_path in message_files[:3]:  # Check last 3 files
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        messages = data if isinstance(data, list) else data.get('messages', [])
                        sample_messages.extend(messages[-sample_size//3:])  # Get recent messages from each file
                        
                        if len(sample_messages) >= sample_size:
                            break
                except Exception as e:
                    logger.debug(f"Error reading message file {file_path}: {e}")
                    continue
            
            return sample_messages[-sample_size:] if sample_messages else []
            
        except Exception as e:
            logger.debug(f"Error getting recent sample messages: {e}")
            return []
    
    def _check_quality_thresholds(self, metrics: QualityMetrics) -> List[QualityAlert]:
        """Check if metrics fall below quality thresholds"""
        alerts = []
        
        checks = [
            ('url_detection_rate', metrics.url_detection_rate, 'URL detection accuracy has dropped'),
            ('code_detection_rate', metrics.code_detection_rate, 'Code snippet detection accuracy has dropped'),
            ('attachment_processing_rate', metrics.attachment_processing_rate, 'Attachment processing has degraded'),
            ('ai_classification_success_rate', metrics.ai_classification_success_rate, 'AI classification is failing'),
            ('noise_filtering_accuracy', metrics.noise_filtering_accuracy, 'Noise filtering is not working properly'),
            ('overall_quality_score', metrics.overall_quality_score, 'Overall system quality has degraded')
        ]
        
        for metric_name, current_value, description in checks:
            threshold = self.thresholds.get(metric_name, 0.5)
            
            if current_value < threshold:
                # Determine severity
                if current_value < threshold * 0.5:
                    severity = "critical"
                elif current_value < threshold * 0.7:
                    severity = "high"
                elif current_value < threshold * 0.85:
                    severity = "medium"
                else:
                    severity = "low"
                
                # Generate recommendation
                recommendation = self._get_quality_recommendation(metric_name, current_value, threshold)
                
                alert = QualityAlert(
                    timestamp=datetime.now(),
                    component="resource_detection",
                    severity=severity,
                    metric=metric_name,
                    current_value=current_value,
                    threshold=threshold,
                    description=description,
                    recommendation=recommendation
                )
                alerts.append(alert)
        
        return alerts
    
    def _get_quality_recommendation(self, metric_name: str, current_value: float, threshold: float) -> str:
        """Generate specific recommendations for quality issues"""
        recommendations = {
            'url_detection_rate': "Check URL regex patterns and domain classification rules",
            'code_detection_rate': "Review code block detection patterns and language identification",
            'attachment_processing_rate': "Verify attachment processing pipeline and error handling",
            'ai_classification_success_rate': "Check OpenAI API connectivity and review classification prompts",
            'noise_filtering_accuracy': "Update noise domain list and filtering logic",
            'overall_quality_score': "Perform comprehensive system health check and component analysis"
        }
        
        base_recommendation = recommendations.get(metric_name, "Investigate system component health")
        
        if current_value == 0.0:
            return f"{base_recommendation}. Component appears to be completely failing."
        elif current_value < threshold * 0.5:
            return f"{base_recommendation}. Critical performance degradation detected."
        else:
            return f"{base_recommendation}. Monitor closely for continued degradation."
    
    def _display_current_status(self, metrics: QualityMetrics, alerts: List[QualityAlert]):
        """Display current monitoring status"""
        timestamp = metrics.timestamp.strftime("%H:%M:%S")
        
        print(f"\n⏰ {timestamp} - Quality Check")
        print("=" * 50)
        
        # Display metrics with color coding
        metric_display = [
            ("URL Detection", metrics.url_detection_rate),
            ("Code Detection", metrics.code_detection_rate),
            ("Attachment Processing", metrics.attachment_processing_rate),
            ("AI Classification", metrics.ai_classification_success_rate),
            ("Noise Filtering", metrics.noise_filtering_accuracy),
            ("Overall Quality", metrics.overall_quality_score)
        ]
        
        for name, value in metric_display:
            status = self._get_status_emoji(value)
            print(f"{status} {name}: {value:.3f}")
        
        # Display alerts
        if alerts:
            print(f"\n🚨 {len(alerts)} Quality Alert(s):")
            for alert in alerts:
                severity_emoji = {"low": "⚠️", "medium": "🔶", "high": "🔴", "critical": "🆘"}
                emoji = severity_emoji.get(alert.severity, "⚠️")
                print(f"  {emoji} {alert.metric}: {alert.current_value:.3f} (threshold: {alert.threshold:.3f})")
        else:
            print("✅ No quality alerts")
    
    def _get_status_emoji(self, value: float) -> str:
        """Get emoji based on metric value"""
        if value >= 0.9:
            return "🟢"
        elif value >= 0.7:
            return "🟡"
        elif value >= 0.5:
            return "🟠"
        else:
            return "🔴"
    
    def _load_historical_data(self):
        """Load historical monitoring data"""
        try:
            data_file = project_root / "data" / "exports" / "quality_monitoring_data.json"
            if data_file.exists():
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    
                    # Load metrics history
                    for metric_data in data.get('metrics_history', []):
                        metrics = QualityMetrics(
                            timestamp=datetime.fromisoformat(metric_data['timestamp']),
                            url_detection_rate=metric_data['url_detection_rate'],
                            code_detection_rate=metric_data['code_detection_rate'],
                            attachment_processing_rate=metric_data['attachment_processing_rate'],
                            ai_classification_success_rate=metric_data['ai_classification_success_rate'],
                            noise_filtering_accuracy=metric_data['noise_filtering_accuracy'],
                            overall_quality_score=metric_data['overall_quality_score']
                        )
                        self.metrics_history.append(metrics)
                    
                    # Load alerts
                    for alert_data in data.get('alerts', []):
                        alert = QualityAlert(
                            timestamp=datetime.fromisoformat(alert_data['timestamp']),
                            component=alert_data['component'],
                            severity=alert_data['severity'],
                            metric=alert_data['metric'],
                            current_value=alert_data['current_value'],
                            threshold=alert_data['threshold'],
                            description=alert_data['description'],
                            recommendation=alert_data['recommendation']
                        )
                        self.alerts.append(alert)
                    
                    logger.info(f"Loaded {len(self.metrics_history)} historical metrics and {len(self.alerts)} alerts")
        except Exception as e:
            logger.debug(f"Could not load historical data: {e}")
    
    def _save_monitoring_data(self):
        """Save monitoring data to file"""
        try:
            data = {
                'last_updated': datetime.now().isoformat(),
                'metrics_history': [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'url_detection_rate': m.url_detection_rate,
                        'code_detection_rate': m.code_detection_rate,
                        'attachment_processing_rate': m.attachment_processing_rate,
                        'ai_classification_success_rate': m.ai_classification_success_rate,
                        'noise_filtering_accuracy': m.noise_filtering_accuracy,
                        'overall_quality_score': m.overall_quality_score
                    }
                    for m in self.metrics_history[-100:]  # Keep last 100 metrics
                ],
                'alerts': [
                    {
                        'timestamp': a.timestamp.isoformat(),
                        'component': a.component,
                        'severity': a.severity,
                        'metric': a.metric,
                        'current_value': a.current_value,
                        'threshold': a.threshold,
                        'description': a.description,
                        'recommendation': a.recommendation
                    }
                    for a in self.alerts[-50:]  # Keep last 50 alerts
                ]
            }
            
            output_dir = project_root / "data" / "exports"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / "quality_monitoring_data.json", 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save monitoring data: {e}")
    
    async def _generate_monitoring_report(self):
        """Generate comprehensive monitoring report"""
        if not self.metrics_history:
            print("📋 No monitoring data to report")
            return
        
        # Calculate summary statistics
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        avg_metrics = {
            'url_detection_rate': statistics.mean([m.url_detection_rate for m in recent_metrics]),
            'code_detection_rate': statistics.mean([m.code_detection_rate for m in recent_metrics]),
            'attachment_processing_rate': statistics.mean([m.attachment_processing_rate for m in recent_metrics]),
            'ai_classification_success_rate': statistics.mean([m.ai_classification_success_rate for m in recent_metrics]),
            'noise_filtering_accuracy': statistics.mean([m.noise_filtering_accuracy for m in recent_metrics]),
            'overall_quality_score': statistics.mean([m.overall_quality_score for m in recent_metrics])
        }
        
        # Count alerts by severity
        alert_counts = {}
        for alert in self.alerts[-20:]:  # Recent alerts
            alert_counts[alert.severity] = alert_counts.get(alert.severity, 0) + 1
        
        # Generate report
        report = {
            'monitoring_summary': {
                'total_checks': len(self.metrics_history),
                'monitoring_period': {
                    'start': self.metrics_history[0].timestamp.isoformat() if self.metrics_history else None,
                    'end': self.metrics_history[-1].timestamp.isoformat() if self.metrics_history else None
                },
                'average_metrics': avg_metrics,
                'alert_summary': alert_counts
            },
            'quality_trends': self._analyze_quality_trends(),
            'recommendations': self._generate_final_recommendations()
        }
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = project_root / "data" / "exports" / f"quality_monitoring_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Display summary
        print(f"\n📋 MONITORING REPORT")
        print("=" * 50)
        print(f"Total Checks: {len(self.metrics_history)}")
        print(f"Average Overall Quality: {avg_metrics['overall_quality_score']:.3f}")
        print(f"Recent Alerts: {sum(alert_counts.values())} ({alert_counts})")
        print(f"Report saved to: {report_file}")
    
    def _analyze_quality_trends(self) -> Dict[str, Any]:
        """Analyze quality trends over time"""
        if len(self.metrics_history) < 2:
            return {"trend": "insufficient_data"}
        
        # Calculate trend for overall quality
        recent_scores = [m.overall_quality_score for m in self.metrics_history[-10:]]
        older_scores = [m.overall_quality_score for m in self.metrics_history[:-10]] if len(self.metrics_history) > 10 else [recent_scores[0]]
        
        recent_avg = statistics.mean(recent_scores)
        older_avg = statistics.mean(older_scores)
        
        trend_direction = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
        trend_magnitude = abs(recent_avg - older_avg)
        
        return {
            "trend": trend_direction,
            "magnitude": trend_magnitude,
            "recent_average": recent_avg,
            "historical_average": older_avg,
            "stability": "stable" if trend_magnitude < 0.1 else "variable"
        }
    
    def _generate_final_recommendations(self) -> List[str]:
        """Generate final recommendations based on monitoring results"""
        recommendations = []
        
        if not self.metrics_history:
            return ["No monitoring data available for recommendations"]
        
        recent_metrics = self.metrics_history[-5:] if len(self.metrics_history) >= 5 else self.metrics_history
        avg_overall = statistics.mean([m.overall_quality_score for m in recent_metrics])
        
        if avg_overall < 0.5:
            recommendations.append("URGENT: System quality is critically low - immediate investigation required")
        elif avg_overall < 0.7:
            recommendations.append("System quality below target - prioritize improvements")
        else:
            recommendations.append("System quality is acceptable - continue monitoring")
        
        # Component-specific recommendations
        avg_ai = statistics.mean([m.ai_classification_success_rate for m in recent_metrics])
        if avg_ai < 0.3:
            recommendations.append("AI classification is severely degraded - check API connectivity and prompts")
        
        avg_url = statistics.mean([m.url_detection_rate for m in recent_metrics])
        if avg_url < 0.8:
            recommendations.append("URL detection needs improvement - review regex patterns")
        
        # Alert-based recommendations
        recent_alerts = [a for a in self.alerts if a.timestamp > datetime.now() - timedelta(hours=24)]
        if len(recent_alerts) > 5:
            recommendations.append("High alert frequency - investigate root causes")
        
        recommendations.append("Consider implementing automated quality recovery mechanisms")
        recommendations.append("Schedule regular evaluation runs to validate improvements")
        
        return recommendations

# CLI interface
async def main():
    """Main monitoring interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Resource Detection Quality Monitor')
    parser.add_argument('--duration', type=int, help='Monitoring duration in minutes (default: continuous)')
    parser.add_argument('--interval', type=int, default=300, help='Check interval in seconds (default: 300)')
    parser.add_argument('--report-only', action='store_true', help='Generate report from existing data only')
    
    args = parser.parse_args()
    
    config = {
        'monitoring_interval': args.interval
    }
    
    monitor = ResourceQualityMonitor(config)
    
    if args.report_only:
        await monitor._generate_monitoring_report()
    else:
        await monitor.start_monitoring(args.duration)

if __name__ == "__main__":
    asyncio.run(main())
