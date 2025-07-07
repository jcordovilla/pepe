"""
Query-Answer Repository System

Comprehensive tracking database for user queries and agent responses
across Discord and Streamlit interfaces with performance monitoring
and validation capabilities.
"""

import asyncio
import json
import sqlite3
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class QueryStatus(Enum):
    """Query processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class AnswerQuality(Enum):
    """Answer quality rating"""
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    POOR = 2
    VERY_POOR = 1


@dataclass
class QueryMetrics:
    """Metrics for query processing"""
    response_time: float
    agents_used: List[str]
    tokens_used: int
    cache_hit: bool
    success: bool
    error_message: Optional[str] = None


@dataclass
class ValidationResult:
    """Validation result for query-answer pair"""
    is_valid: bool
    quality_score: float
    relevance_score: float
    completeness_score: float
    accuracy_score: float
    issues: List[str]
    suggestions: List[str]


class QueryAnswerRepository:
    """
    Comprehensive repository for tracking and analyzing query-answer pairs.
    
    Features:
    - Cross-platform tracking (Discord + Streamlit)
    - Performance metrics and analytics
    - Quality assessment and validation
    - Trend analysis and reporting
    - Export capabilities for further analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get("analytics_db_path", "data/analytics.db")
        self.enable_validation = config.get("enable_validation", True)
        self.auto_validate = config.get("auto_validate", True)
        self.retention_days = config.get("retention_days", 90)
        
        self._init_database()
        logger.info("Query-Answer Repository initialized")
    
    def _init_database(self):
        """Initialize the analytics database with comprehensive schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main query tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT NOT NULL,
                user_id TEXT NOT NULL,
                platform TEXT NOT NULL,  -- 'discord' or 'streamlit'
                channel_id TEXT,         -- Discord channel or Streamlit session
                query_text TEXT NOT NULL,
                answer_text TEXT NOT NULL,
                query_timestamp DATETIME NOT NULL,
                response_timestamp DATETIME NOT NULL,
                status TEXT NOT NULL,    -- QueryStatus enum value
                response_time REAL NOT NULL,
                tokens_used INTEGER DEFAULT 0,
                agents_used TEXT,        -- JSON array of agent names
                cache_hit BOOLEAN DEFAULT FALSE,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                context TEXT,           -- JSON context information
                metadata TEXT,          -- JSON metadata
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit TEXT,
                recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (query_id) REFERENCES query_answers (id)
            )
        """)
        
        # Quality validation table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS answer_validation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER NOT NULL,
                validator_type TEXT NOT NULL,  -- 'auto', 'human', 'ai'
                quality_score REAL,           -- 1-5 rating
                relevance_score REAL,         -- 0-1 score
                completeness_score REAL,      -- 0-1 score  
                accuracy_score REAL,          -- 0-1 score
                overall_rating TEXT,          -- AnswerQuality enum value
                issues TEXT,                  -- JSON array of issues
                suggestions TEXT,             -- JSON array of suggestions
                validated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                validator_info TEXT,          -- JSON info about validator
                FOREIGN KEY (query_id) REFERENCES query_answers (id)
            )
        """)
        
        # User feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER NOT NULL,
                user_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL,  -- 'thumbs_up', 'thumbs_down', 'rating', 'comment'
                rating INTEGER,               -- 1-5 rating if applicable
                comment TEXT,
                submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (query_id) REFERENCES query_answers (id)
            )
        """)
        
        # Trend analysis table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trend_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date_bucket TEXT NOT NULL,    -- YYYY-MM-DD or YYYY-MM-DD-HH
                platform TEXT NOT NULL,
                total_queries INTEGER DEFAULT 0,
                successful_queries INTEGER DEFAULT 0,
                avg_response_time REAL DEFAULT 0,
                avg_quality_score REAL DEFAULT 0,
                cache_hit_rate REAL DEFAULT 0,
                unique_users INTEGER DEFAULT 0,
                top_query_types TEXT,         -- JSON array
                performance_issues TEXT,      -- JSON array
                calculated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # System performance snapshots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                memory_usage REAL,
                cpu_usage REAL,
                disk_usage REAL,
                active_connections INTEGER,
                cache_size INTEGER,
                database_size INTEGER,
                vector_store_size INTEGER,
                system_status TEXT,
                notes TEXT
            )
        """)
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_qa_user_id ON query_answers(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_qa_platform ON query_answers(platform)",
            "CREATE INDEX IF NOT EXISTS idx_qa_timestamp ON query_answers(query_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_qa_status ON query_answers(status)",
            "CREATE INDEX IF NOT EXISTS idx_qa_success ON query_answers(success)",
            "CREATE INDEX IF NOT EXISTS idx_qa_query_hash ON query_answers(query_hash)",
            "CREATE INDEX IF NOT EXISTS idx_validation_quality ON answer_validation(quality_score)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_user ON user_feedback(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_trends_date ON trend_analysis(date_bucket)",
            "CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON system_snapshots(timestamp)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        conn.close()
    
    async def record_query_answer(
        self,
        user_id: str,
        platform: str,
        query_text: str,
        answer_text: str,
        metrics: QueryMetrics,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        channel_id: Optional[str] = None
    ) -> int:
        """
        Record a new query-answer pair with comprehensive tracking.
        
        Args:
            user_id: User identifier
            platform: Platform name ('discord' or 'streamlit')
            query_text: User's query
            answer_text: System's response
            metrics: Performance metrics
            context: Additional context information
            metadata: Additional metadata
            channel_id: Channel/session identifier
            
        Returns:
            ID of the recorded entry
        """
        query_hash = hashlib.md5(query_text.encode()).hexdigest()
        query_timestamp = datetime.utcnow()
        response_timestamp = query_timestamp + timedelta(seconds=metrics.response_time)
        
        status = QueryStatus.COMPLETED if metrics.success else QueryStatus.FAILED
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO query_answers (
                    query_hash, user_id, platform, channel_id, query_text,
                    answer_text, query_timestamp, response_timestamp, status,
                    response_time, tokens_used, agents_used, cache_hit,
                    success, error_message, context, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query_hash, user_id, platform, channel_id, query_text,
                answer_text, query_timestamp.isoformat(), response_timestamp.isoformat(),
                status.value, metrics.response_time, metrics.tokens_used,
                json.dumps(metrics.agents_used), metrics.cache_hit, metrics.success,
                metrics.error_message, json.dumps(context or {}),
                json.dumps(metadata or {})
            ))
            
            query_id = cursor.lastrowid
            
            # Record additional performance metrics
            await self._record_performance_metrics(cursor, query_id, metrics)
            
            conn.commit()
            
            # Trigger auto-validation if enabled
            if self.auto_validate and metrics.success:
                await self._auto_validate_answer(query_id, query_text, answer_text)
            
            logger.info(f"Recorded query-answer pair {query_id} for user {user_id}")
            return query_id
            
        except Exception as e:
            logger.error(f"Error recording query-answer: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    async def _record_performance_metrics(
        self, 
        cursor: sqlite3.Cursor, 
        query_id: int, 
        metrics: QueryMetrics
    ):
        """Record detailed performance metrics"""
        performance_data = [
            ("response_time", metrics.response_time, "seconds"),
            ("tokens_used", metrics.tokens_used, "tokens"),
            ("agents_count", len(metrics.agents_used), "count"),
            ("cache_hit", 1.0 if metrics.cache_hit else 0.0, "boolean")
        ]
        
        for metric_name, value, unit in performance_data:
            cursor.execute("""
                INSERT INTO performance_metrics (query_id, metric_name, metric_value, metric_unit)
                VALUES (?, ?, ?, ?)
            """, (query_id, metric_name, value, unit))
    
    async def _auto_validate_answer(self, query_id: int, query_text: str, answer_text: str):
        """Automatically validate answer quality using basic heuristics"""
        try:
            # Basic validation heuristics
            relevance_score = min(1.0, len(answer_text) / max(100, len(query_text) * 10))
            completeness_score = min(1.0, len(answer_text) / 200)  # Assume 200 chars is complete
            accuracy_score = 0.8  # Default assumption for successful responses
            
            quality_score = (relevance_score + completeness_score + accuracy_score) / 3 * 5
            
            issues = []
            suggestions = []
            
            if len(answer_text) < 50:
                issues.append("Answer appears too short")
                suggestions.append("Consider providing more detailed explanation")
            
            if len(answer_text) > 2000:
                issues.append("Answer appears too long")
                suggestions.append("Consider summarizing key points")
            
            validation_result = ValidationResult(
                is_valid=True,
                quality_score=quality_score,
                relevance_score=relevance_score,
                completeness_score=completeness_score,
                accuracy_score=accuracy_score,
                issues=issues,
                suggestions=suggestions
            )
            
            await self.record_validation(query_id, "auto", validation_result)
            
        except Exception as e:
            logger.warning(f"Auto-validation failed for query {query_id}: {e}")
    
    async def record_validation(
        self,
        query_id: int,
        validator_type: str,
        result: ValidationResult,
        validator_info: Optional[Dict[str, Any]] = None
    ):
        """Record answer validation results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            overall_rating = AnswerQuality.EXCELLENT
            if result.quality_score < 2:
                overall_rating = AnswerQuality.VERY_POOR
            elif result.quality_score < 3:
                overall_rating = AnswerQuality.POOR
            elif result.quality_score < 4:
                overall_rating = AnswerQuality.AVERAGE
            elif result.quality_score < 4.5:
                overall_rating = AnswerQuality.GOOD
            
            cursor.execute("""
                INSERT INTO answer_validation (
                    query_id, validator_type, quality_score, relevance_score,
                    completeness_score, accuracy_score, overall_rating,
                    issues, suggestions, validator_info
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query_id, validator_type, result.quality_score,
                result.relevance_score, result.completeness_score,
                result.accuracy_score, overall_rating.value,
                json.dumps(result.issues), json.dumps(result.suggestions),
                json.dumps(validator_info or {})
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error recording validation: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    async def record_user_feedback(
        self,
        query_id: int,
        user_id: str,
        feedback_type: str,
        rating: Optional[int] = None,
        comment: Optional[str] = None
    ):
        """Record user feedback for a query-answer pair"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO user_feedback (query_id, user_id, feedback_type, rating, comment)
                VALUES (?, ?, ?, ?, ?)
            """, (query_id, user_id, feedback_type, rating, comment))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error recording user feedback: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    async def get_query_history(
        self,
        user_id: Optional[str] = None,
        platform: Optional[str] = None,
        hours_back: int = 24,
        limit: int = 100,
        include_validation: bool = True
    ) -> List[Dict[str, Any]]:
        """Get query history with optional filtering"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            base_query = """
                SELECT qa.*, av.quality_score, av.overall_rating
                FROM query_answers qa
                LEFT JOIN answer_validation av ON qa.id = av.query_id AND av.validator_type = 'auto'
                WHERE qa.query_timestamp > ?
            """
            
            params = [
                (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()
            ]
            
            if user_id:
                base_query += " AND qa.user_id = ?"
                params.append(user_id)
            
            if platform:
                base_query += " AND qa.platform = ?"
                params.append(platform)
            
            base_query += " ORDER BY qa.query_timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(base_query, params)
            rows = cursor.fetchall()
            
            columns = [desc[0] for desc in cursor.description]
            history = []
            
            for row in rows:
                entry = dict(zip(columns, row))
                
                # Parse JSON fields
                entry["agents_used"] = json.loads(entry.get("agents_used", "[]"))
                entry["context"] = json.loads(entry.get("context", "{}"))
                entry["metadata"] = json.loads(entry.get("metadata", "{}"))
                
                history.append(entry)
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting query history: {e}")
            return []
        finally:
            conn.close()
    
    async def get_performance_analytics(
        self,
        hours_back: int = 24,
        platform: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cutoff_time = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()
            
            # Basic statistics
            query = """
                SELECT 
                    COUNT(*) as total_queries,
                    AVG(response_time) as avg_response_time,
                    AVG(tokens_used) as avg_tokens_used,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_queries,
                    SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cache_hits,
                    COUNT(DISTINCT user_id) as unique_users
                FROM query_answers
                WHERE query_timestamp > ?
            """
            
            params = [cutoff_time]
            if platform:
                query += " AND platform = ?"
                params.append(platform)
            
            cursor.execute(query, params)
            stats = cursor.fetchone()
            
            # Quality metrics
            quality_query = """
                SELECT AVG(av.quality_score) as avg_quality_score
                FROM answer_validation av
                JOIN query_answers qa ON av.query_id = qa.id
                WHERE qa.query_timestamp > ?
            """
            
            if platform:
                quality_query += " AND qa.platform = ?"
            
            cursor.execute(quality_query, params)
            quality_stats = cursor.fetchone()
            
            # Response time distribution
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN response_time < 1 THEN 'fast'
                        WHEN response_time < 3 THEN 'medium'
                        ELSE 'slow'
                    END as speed_category,
                    COUNT(*) as count
                FROM query_answers
                WHERE query_timestamp > ?
                GROUP BY speed_category
            """, [cutoff_time])
            
            speed_distribution = dict(cursor.fetchall())
            
            total_queries = stats[0] if stats[0] else 0
            success_rate = (stats[3] / total_queries * 100) if total_queries > 0 else 0
            cache_hit_rate = (stats[5] / total_queries * 100) if total_queries > 0 else 0
            
            return {
                "period_hours": hours_back,
                "platform": platform or "all",
                "total_queries": total_queries,
                "successful_queries": stats[3] if stats else 0,
                "success_rate": round(success_rate, 2),
                "avg_response_time": round(stats[1] if stats[1] else 0, 3),
                "avg_tokens_used": round(stats[2] if stats[2] else 0, 1),
                "cache_hit_rate": round(cache_hit_rate, 2),
                "unique_users": stats[6] if stats else 0,
                "avg_quality_score": round(quality_stats[0] if quality_stats[0] else 0, 2),
                "speed_distribution": speed_distribution,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance analytics: {e}")
            return {}
        finally:
            conn.close()
    
    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get analytics for a specific user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # User statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    AVG(response_time) as avg_response_time,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_queries,
                    MIN(query_timestamp) as first_query,
                    MAX(query_timestamp) as last_query,
                    COUNT(DISTINCT platform) as platforms_used
                FROM query_answers
                WHERE user_id = ?
            """, (user_id,))
            
            stats = cursor.fetchone()
            
            # Query patterns
            cursor.execute("""
                SELECT platform, COUNT(*) as count
                FROM query_answers
                WHERE user_id = ?
                GROUP BY platform
            """, (user_id,))
            
            platform_usage = dict(cursor.fetchall())
            
            # Recent quality scores
            cursor.execute("""
                SELECT AVG(av.quality_score) as avg_quality
                FROM answer_validation av
                JOIN query_answers qa ON av.query_id = qa.id
                WHERE qa.user_id = ?
            """, (user_id,))
            
            quality_result = cursor.fetchone()
            
            total_queries = stats[0] if stats[0] else 0
            success_rate = (stats[2] / total_queries * 100) if total_queries > 0 else 0
            
            return {
                "user_id": user_id,
                "total_queries": total_queries,
                "successful_queries": stats[2] if stats else 0,
                "success_rate": round(success_rate, 2),
                "avg_response_time": round(stats[1] if stats[1] else 0, 3),
                "first_query": stats[3],
                "last_query": stats[4],
                "platforms_used": platform_usage,
                "avg_quality_score": round(quality_result[0] if quality_result[0] else 0, 2),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting user analytics: {e}")
            return {}
        finally:
            conn.close()
    
    async def record_system_snapshot(self, system_metrics: Dict[str, Any]):
        """Record system performance snapshot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO system_snapshots (
                    memory_usage, cpu_usage, disk_usage, active_connections,
                    cache_size, database_size, vector_store_size, system_status, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                system_metrics.get("memory_usage"),
                system_metrics.get("cpu_usage"),
                system_metrics.get("disk_usage"),
                system_metrics.get("active_connections"),
                system_metrics.get("cache_size"),
                system_metrics.get("database_size"),
                system_metrics.get("vector_store_size"),
                system_metrics.get("system_status", "unknown"),
                system_metrics.get("notes")
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error recording system snapshot: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    async def cleanup_old_data(self):
        """Clean up old data based on retention policy"""
        if self.retention_days <= 0:
            return
        
        cutoff_date = (datetime.utcnow() - timedelta(days=self.retention_days)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Clean up old query answers and related data
            cursor.execute("""
                DELETE FROM query_answers WHERE query_timestamp < ?
            """, (cutoff_date,))
            
            # Clean up orphaned validation records
            cursor.execute("""
                DELETE FROM answer_validation 
                WHERE query_id NOT IN (SELECT id FROM query_answers)
            """)
            
            # Clean up orphaned feedback records
            cursor.execute("""
                DELETE FROM user_feedback 
                WHERE query_id NOT IN (SELECT id FROM query_answers)
            """)
            
            # Clean up orphaned performance metrics
            cursor.execute("""
                DELETE FROM performance_metrics 
                WHERE query_id NOT IN (SELECT id FROM query_answers)
            """)
            
            # Clean up old system snapshots
            cursor.execute("""
                DELETE FROM system_snapshots WHERE timestamp < ?
            """, (cutoff_date,))
            
            conn.commit()
            logger.info(f"Cleaned up data older than {self.retention_days} days")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    async def export_data(
        self,
        format_type: str = "json",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Export analytics data for external analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            query = "SELECT * FROM query_answers WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND query_timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND query_timestamp <= ?"
                params.append(end_date.isoformat())
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            data = []
            for row in rows:
                entry = dict(zip(columns, row))
                # Parse JSON fields
                entry["agents_used"] = json.loads(entry.get("agents_used", "[]"))
                entry["context"] = json.loads(entry.get("context", "{}"))
                entry["metadata"] = json.loads(entry.get("metadata", "{}"))
                data.append(entry)
            
            return {
                "format": format_type,
                "exported_at": datetime.utcnow().isoformat(),
                "total_records": len(data),
                "data": data
            }
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return {}
        finally:
            conn.close()
