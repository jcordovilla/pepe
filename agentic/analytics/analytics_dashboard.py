"""
Analytics Dashboard

Interactive dashboard for visualizing query-answer analytics, performance metrics,
and validation results across Discord and Streamlit interfaces.
"""

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class AnalyticsDashboard:
    """
    Comprehensive analytics dashboard for query-answer tracking system.
    
    Features:
    - Interactive visualizations using Plotly
    - Real-time performance monitoring
    - Cross-platform analytics (Discord + Streamlit)
    - Quality trend analysis
    - User behavior insights
    - Export capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.query_repository = None
        self.performance_monitor = None
        self.validation_system = None
        
        # Dashboard configuration
        self.default_time_range = config.get("default_time_range", 24)  # hours
        self.chart_height = config.get("chart_height", 400)
        self.color_scheme = config.get("color_scheme", "plotly")
        
        logger.info("Analytics Dashboard initialized")
    
    def set_components(self, query_repository, performance_monitor, validation_system):
        """Set component references for data access"""
        self.query_repository = query_repository
        self.performance_monitor = performance_monitor
        self.validation_system = validation_system
    
    async def generate_overview_dashboard(
        self,
        hours_back: int = 24,
        platform: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive overview dashboard"""
        try:
            # Get performance analytics
            performance_data = await self.query_repository.get_performance_analytics(
                hours_back=hours_back,
                platform=platform
            )
            
            # Get system status
            system_status = self.performance_monitor.get_current_status()
            
            # Get recent queries
            recent_queries = await self.query_repository.get_query_history(
                hours_back=hours_back,
                platform=platform,
                limit=100
            )
            
            # Generate visualizations
            charts = await self._generate_overview_charts(
                performance_data, recent_queries, hours_back
            )
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "time_range_hours": hours_back,
                "platform": platform or "all",
                "performance_data": performance_data,
                "system_status": system_status,
                "total_queries": len(recent_queries),
                "charts": charts
            }
            
        except Exception as e:
            logger.error(f"Error generating overview dashboard: {e}")
            return {"error": str(e)}
    
    async def _generate_overview_charts(
        self,
        performance_data: Dict[str, Any],
        recent_queries: List[Dict[str, Any]],
        hours_back: int
    ) -> Dict[str, Any]:
        """Generate overview charts"""
        charts = {}
        
        # 1. Query Volume Over Time
        charts["query_volume"] = self._create_query_volume_chart(recent_queries, hours_back)
        
        # 2. Response Time Distribution
        charts["response_time"] = self._create_response_time_chart(recent_queries)
        
        # 3. Success Rate Chart
        charts["success_rate"] = self._create_success_rate_chart(performance_data)
        
        # 4. Platform Usage
        charts["platform_usage"] = self._create_platform_usage_chart(recent_queries)
        
        # 5. Quality Score Distribution
        charts["quality_scores"] = self._create_quality_score_chart(recent_queries)
        
        return charts
    
    def _create_query_volume_chart(
        self,
        recent_queries: List[Dict[str, Any]],
        hours_back: int
    ) -> Dict[str, Any]:
        """Create query volume over time chart"""
        try:
            if not recent_queries:
                return {"error": "No data available"}
            
            # Convert to DataFrame
            df = pd.DataFrame(recent_queries)
            df['query_timestamp'] = pd.to_datetime(df['query_timestamp'])
            
            # Group by hour
            df['hour'] = df['query_timestamp'].dt.floor('H')
            hourly_counts = df.groupby('hour').size().reset_index(name='count')
            
            # Create line chart
            fig = px.line(
                hourly_counts,
                x='hour',
                y='count',
                title='Query Volume Over Time',
                labels={'hour': 'Time', 'count': 'Number of Queries'}
            )
            
            fig.update_layout(height=self.chart_height)
            
            return {"chart": fig.to_json(), "type": "line"}
            
        except Exception as e:
            logger.error(f"Error creating query volume chart: {e}")
            return {"error": str(e)}
    
    def _create_response_time_chart(
        self,
        recent_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create response time distribution chart"""
        try:
            if not recent_queries:
                return {"error": "No data available"}
            
            # Extract response times
            response_times = [q.get('response_time', 0) for q in recent_queries]
            
            # Create histogram
            fig = px.histogram(
                x=response_times,
                nbins=20,
                title='Response Time Distribution',
                labels={'x': 'Response Time (seconds)', 'y': 'Frequency'}
            )
            
            # Add average line
            avg_time = sum(response_times) / len(response_times)
            fig.add_vline(
                x=avg_time,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Avg: {avg_time:.2f}s"
            )
            
            fig.update_layout(height=self.chart_height)
            
            return {"chart": fig.to_json(), "type": "histogram"}
            
        except Exception as e:
            logger.error(f"Error creating response time chart: {e}")
            return {"error": str(e)}
    
    def _create_success_rate_chart(
        self,
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create success rate gauge chart"""
        try:
            success_rate = performance_data.get('success_rate', 0)
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=success_rate,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Success Rate (%)"},
                delta={'reference': 95},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 80], 'color': "lightgray"},
                        {'range': [80, 95], 'color': "yellow"},
                        {'range': [95, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=self.chart_height)
            
            return {"chart": fig.to_json(), "type": "gauge"}
            
        except Exception as e:
            logger.error(f"Error creating success rate chart: {e}")
            return {"error": str(e)}
    
    def _create_platform_usage_chart(
        self,
        recent_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create platform usage pie chart"""
        try:
            if not recent_queries:
                return {"error": "No data available"}
            
            # Count queries by platform
            platform_counts = {}
            for query in recent_queries:
                platform = query.get('platform', 'unknown')
                platform_counts[platform] = platform_counts.get(platform, 0) + 1
            
            # Create pie chart
            fig = px.pie(
                values=list(platform_counts.values()),
                names=list(platform_counts.keys()),
                title='Queries by Platform'
            )
            
            fig.update_layout(height=self.chart_height)
            
            return {"chart": fig.to_json(), "type": "pie"}
            
        except Exception as e:
            logger.error(f"Error creating platform usage chart: {e}")
            return {"error": str(e)}
    
    def _create_quality_score_chart(
        self,
        recent_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create quality score distribution chart"""
        try:
            # Extract quality scores (only for queries with validation)
            quality_scores = []
            for query in recent_queries:
                score = query.get('quality_score')
                if score is not None:
                    quality_scores.append(score)
            
            if not quality_scores:
                return {"error": "No quality scores available"}
            
            # Create box plot
            fig = px.box(
                y=quality_scores,
                title='Quality Score Distribution',
                labels={'y': 'Quality Score (1-5)'}
            )
            
            fig.update_layout(height=self.chart_height)
            
            return {"chart": fig.to_json(), "type": "box"}
            
        except Exception as e:
            logger.error(f"Error creating quality score chart: {e}")
            return {"error": str(e)}
    
    async def generate_performance_dashboard(
        self,
        hours_back: int = 24
    ) -> Dict[str, Any]:
        """Generate performance-focused dashboard"""
        try:
            # Get performance trends
            performance_trends = self.performance_monitor.get_performance_trends(hours_back)
            
            # Get current system status
            system_status = self.performance_monitor.get_current_status()
            
            # Get alerts summary
            alerts_summary = self.performance_monitor.get_alerts_summary(hours_back)
            
            # Generate performance charts
            charts = await self._generate_performance_charts(
                performance_trends, system_status, alerts_summary
            )
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "time_range_hours": hours_back,
                "performance_trends": performance_trends,
                "system_status": system_status,
                "alerts_summary": alerts_summary,
                "charts": charts
            }
            
        except Exception as e:
            logger.error(f"Error generating performance dashboard: {e}")
            return {"error": str(e)}
    
    async def _generate_performance_charts(
        self,
        performance_trends: Dict[str, Any],
        system_status: Dict[str, Any],
        alerts_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate performance-specific charts"""
        charts = {}
        
        # 1. System Health Overview
        charts["system_health"] = self._create_system_health_chart(system_status)
        
        # 2. Performance Trends
        charts["performance_trends"] = self._create_performance_trends_chart(performance_trends)
        
        # 3. Alerts Timeline
        charts["alerts_timeline"] = self._create_alerts_chart(alerts_summary)
        
        return charts
    
    def _create_system_health_chart(
        self,
        system_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create system health status chart"""
        try:
            metrics = system_status.get('metrics', {})
            
            # Create radar chart for system health
            categories = ['Response Time', 'Success Rate', 'Memory Usage', 
                         'CPU Usage', 'Cache Hit Rate']
            
            values = [
                min(5, 5 - metrics.get('response_time', 0)),  # Invert response time
                metrics.get('success_rate', 0) / 20,  # Scale to 0-5
                5 - (metrics.get('memory_usage', 0) / 20),  # Invert memory usage
                5 - (metrics.get('cpu_usage', 0) / 20),  # Invert CPU usage
                metrics.get('cache_hit_rate', 0) / 20  # Scale to 0-5
            ]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Current Status'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5]
                    )
                ),
                title="System Health Overview",
                height=self.chart_height
            )
            
            return {"chart": fig.to_json(), "type": "radar"}
            
        except Exception as e:
            logger.error(f"Error creating system health chart: {e}")
            return {"error": str(e)}
    
    def _create_performance_trends_chart(
        self,
        performance_trends: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create performance trends chart"""
        try:
            trends = performance_trends.get('trends', {})
            
            # Create subplots for different metrics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Response Time', 'Success Rate', 'Memory Usage', 'Cache Performance'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Mock time series data (in real implementation, would use actual historical data)
            hours = list(range(24))
            
            # Response time trend
            response_times = [trends.get('response_time', {}).get('avg', 1.2) + 
                            (i % 5 - 2) * 0.1 for i in hours]
            fig.add_trace(
                go.Scatter(x=hours, y=response_times, name="Response Time"),
                row=1, col=1
            )
            
            # Success rate trend
            success_rates = [trends.get('success_rate', {}).get('avg', 98.5) + 
                           (i % 3 - 1) * 0.5 for i in hours]
            fig.add_trace(
                go.Scatter(x=hours, y=success_rates, name="Success Rate"),
                row=1, col=2
            )
            
            # Memory usage trend
            memory_usage = [trends.get('memory_usage', {}).get('avg', 65) + 
                          (i % 7 - 3) * 2 for i in hours]
            fig.add_trace(
                go.Scatter(x=hours, y=memory_usage, name="Memory Usage"),
                row=2, col=1
            )
            
            # Cache performance (mock data)
            cache_performance = [75 + (i % 4 - 2) * 3 for i in hours]
            fig.add_trace(
                go.Scatter(x=hours, y=cache_performance, name="Cache Hit Rate"),
                row=2, col=2
            )
            
            fig.update_layout(
                title="Performance Trends (24h)",
                height=self.chart_height * 1.5,
                showlegend=False
            )
            
            return {"chart": fig.to_json(), "type": "multi_line"}
            
        except Exception as e:
            logger.error(f"Error creating performance trends chart: {e}")
            return {"error": str(e)}
    
    def _create_alerts_chart(
        self,
        alerts_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create alerts summary chart"""
        try:
            by_level = alerts_summary.get('by_level', {})
            
            if not by_level:
                return {"message": "No alerts in time period"}
            
            # Create bar chart for alerts by level
            fig = px.bar(
                x=list(by_level.keys()),
                y=list(by_level.values()),
                title='Alerts by Severity Level',
                labels={'x': 'Alert Level', 'y': 'Count'},
                color=list(by_level.keys()),
                color_discrete_map={
                    'critical': 'red',
                    'error': 'orange',
                    'warning': 'yellow',
                    'info': 'blue'
                }
            )
            
            fig.update_layout(height=self.chart_height)
            
            return {"chart": fig.to_json(), "type": "bar"}
            
        except Exception as e:
            logger.error(f"Error creating alerts chart: {e}")
            return {"error": str(e)}
    
    async def generate_user_analytics(
        self,
        user_id: Optional[str] = None,
        hours_back: int = 168  # 1 week
    ) -> Dict[str, Any]:
        """Generate user-focused analytics"""
        try:
            if user_id:
                # Single user analytics
                user_data = await self.query_repository.get_user_analytics(user_id)
                user_queries = await self.query_repository.get_query_history(
                    user_id=user_id,
                    hours_back=hours_back,
                    limit=200
                )
                
                charts = await self._generate_user_charts(user_data, user_queries)
                
                return {
                    "timestamp": datetime.utcnow().isoformat(),
                    "user_id": user_id,
                    "time_range_hours": hours_back,
                    "user_data": user_data,
                    "total_queries": len(user_queries),
                    "charts": charts
                }
            else:
                # All users analytics
                all_queries = await self.query_repository.get_query_history(
                    hours_back=hours_back,
                    limit=1000
                )
                
                charts = await self._generate_users_overview_charts(all_queries)
                
                return {
                    "timestamp": datetime.utcnow().isoformat(),
                    "time_range_hours": hours_back,
                    "total_queries": len(all_queries),
                    "charts": charts
                }
            
        except Exception as e:
            logger.error(f"Error generating user analytics: {e}")
            return {"error": str(e)}
    
    async def _generate_user_charts(
        self,
        user_data: Dict[str, Any],
        user_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate charts for individual user analytics"""
        charts = {}
        
        # 1. User query activity over time
        charts["user_activity"] = self._create_user_activity_chart(user_queries)
        
        # 2. User query topics/patterns
        charts["query_patterns"] = self._create_query_patterns_chart(user_queries)
        
        # 3. User satisfaction scores
        charts["satisfaction"] = self._create_user_satisfaction_chart(user_queries)
        
        return charts
    
    def _create_user_activity_chart(
        self,
        user_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create user activity over time chart"""
        try:
            if not user_queries:
                return {"error": "No user queries available"}
            
            # Convert to DataFrame and group by day
            df = pd.DataFrame(user_queries)
            df['query_timestamp'] = pd.to_datetime(df['query_timestamp'])
            df['date'] = df['query_timestamp'].dt.date
            
            daily_counts = df.groupby('date').size().reset_index(name='count')
            
            fig = px.bar(
                daily_counts,
                x='date',
                y='count',
                title='User Query Activity Over Time',
                labels={'date': 'Date', 'count': 'Number of Queries'}
            )
            
            fig.update_layout(height=self.chart_height)
            
            return {"chart": fig.to_json(), "type": "bar"}
            
        except Exception as e:
            logger.error(f"Error creating user activity chart: {e}")
            return {"error": str(e)}
    
    def _create_query_patterns_chart(
        self,
        user_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create query patterns analysis chart"""
        try:
            if not user_queries:
                return {"error": "No user queries available"}
            
            # Analyze query lengths
            query_lengths = [len(q.get('query_text', '')) for q in user_queries]
            
            fig = px.histogram(
                x=query_lengths,
                nbins=15,
                title='User Query Length Distribution',
                labels={'x': 'Query Length (characters)', 'y': 'Frequency'}
            )
            
            fig.update_layout(height=self.chart_height)
            
            return {"chart": fig.to_json(), "type": "histogram"}
            
        except Exception as e:
            logger.error(f"Error creating query patterns chart: {e}")
            return {"error": str(e)}
    
    def _create_user_satisfaction_chart(
        self,
        user_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create user satisfaction chart based on quality scores"""
        try:
            # Extract quality scores for this user
            quality_scores = []
            for query in user_queries:
                score = query.get('quality_score')
                if score is not None:
                    quality_scores.append(score)
            
            if not quality_scores:
                return {"message": "No quality scores available for this user"}
            
            # Create gauge for average satisfaction
            avg_score = sum(quality_scores) / len(quality_scores)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Average Answer Quality"},
                gauge={
                    'axis': {'range': [None, 5]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 2], 'color': "red"},
                        {'range': [2, 3.5], 'color': "yellow"},
                        {'range': [3.5, 5], 'color': "green"}
                    ]
                }
            ))
            
            fig.update_layout(height=self.chart_height)
            
            return {"chart": fig.to_json(), "type": "gauge"}
            
        except Exception as e:
            logger.error(f"Error creating user satisfaction chart: {e}")
            return {"error": str(e)}
    
    async def _generate_users_overview_charts(
        self,
        all_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate overview charts for all users"""
        charts = {}
        
        # 1. Top users by query volume
        charts["top_users"] = self._create_top_users_chart(all_queries)
        
        # 2. User engagement patterns
        charts["engagement_patterns"] = self._create_engagement_patterns_chart(all_queries)
        
        return charts
    
    def _create_top_users_chart(
        self,
        all_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create top users by query volume chart"""
        try:
            if not all_queries:
                return {"error": "No queries available"}
            
            # Count queries by user
            user_counts = {}
            for query in all_queries:
                user_id = query.get('user_id', 'unknown')
                user_counts[user_id] = user_counts.get(user_id, 0) + 1
            
            # Get top 10 users
            top_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            if not top_users:
                return {"error": "No user data available"}
            
            users, counts = zip(*top_users)
            
            fig = px.bar(
                x=list(counts),
                y=list(users),
                orientation='h',
                title='Top Users by Query Volume',
                labels={'x': 'Number of Queries', 'y': 'User ID'}
            )
            
            fig.update_layout(height=self.chart_height)
            
            return {"chart": fig.to_json(), "type": "horizontal_bar"}
            
        except Exception as e:
            logger.error(f"Error creating top users chart: {e}")
            return {"error": str(e)}
    
    def _create_engagement_patterns_chart(
        self,
        all_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create user engagement patterns chart"""
        try:
            if not all_queries:
                return {"error": "No queries available"}
            
            # Analyze queries by hour of day
            df = pd.DataFrame(all_queries)
            df['query_timestamp'] = pd.to_datetime(df['query_timestamp'])
            df['hour'] = df['query_timestamp'].dt.hour
            
            hourly_counts = df.groupby('hour').size().reset_index(name='count')
            
            fig = px.line(
                hourly_counts,
                x='hour',
                y='count',
                title='Query Volume by Hour of Day',
                labels={'hour': 'Hour of Day', 'count': 'Number of Queries'}
            )
            
            fig.update_layout(height=self.chart_height)
            
            return {"chart": fig.to_json(), "type": "line"}
            
        except Exception as e:
            logger.error(f"Error creating engagement patterns chart: {e}")
            return {"error": str(e)}
    
    async def export_dashboard_data(
        self,
        dashboard_type: str,
        format_type: str = "json",
        **kwargs
    ) -> Dict[str, Any]:
        """Export dashboard data in specified format"""
        try:
            if dashboard_type == "overview":
                data = await self.generate_overview_dashboard(**kwargs)
            elif dashboard_type == "performance":
                data = await self.generate_performance_dashboard(**kwargs)
            elif dashboard_type == "user_analytics":
                data = await self.generate_user_analytics(**kwargs)
            else:
                return {"error": f"Unknown dashboard type: {dashboard_type}"}
            
            if format_type == "json":
                return data
            elif format_type == "csv":
                # Convert to CSV format (simplified)
                return {"error": "CSV export not yet implemented"}
            else:
                return {"error": f"Unknown format type: {format_type}"}
            
        except Exception as e:
            logger.error(f"Error exporting dashboard data: {e}")
            return {"error": str(e)}
