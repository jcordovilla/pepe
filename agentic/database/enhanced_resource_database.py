"""
Enhanced Resource Database Integration

Comprehensive resource database that integrates with vector store and provides
unified resource management, validation tracking, and query optimization.
"""

import sqlite3
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib

from ..services.enhanced_resource_detector import ResourceMetadata, ResourceType, ResourceQuality
from ..services.resource_validation_pipeline import ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


@dataclass
class ResourceRecord:
    """Enhanced resource record with vector integration"""
    id: str
    embedding_id: Optional[str]  # Link to ChromaDB embedding
    message_id: str
    channel_id: str
    resource_type: str
    url: Optional[str]
    title: Optional[str]
    description: Optional[str]
    quality_score: float
    quality_level: str
    validation_status: str
    validation_timestamp: Optional[datetime]
    content_hash: Optional[str]
    domain: Optional[str]
    file_size: Optional[int]
    content_type: Optional[str]
    language: Optional[str]
    keywords: List[str]
    topics: List[str]
    author: Optional[str]
    created_at: datetime
    updated_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    is_active: bool = True


class EnhancedResourceDatabase:
    """
    Enhanced resource database with vector store integration
    """
    
    def __init__(self, db_path: str = "data/enhanced_resources.db"):
        self.db_path = db_path
        self.connection = None
        
        # Performance optimization settings
        self.batch_size = 100
        self.cache_ttl = 3600
        
        logger.info(f"🗄️ Enhanced resource database initialized: {db_path}")
    
    async def initialize(self):
        """Initialize database with comprehensive schema"""
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        
        # Enable performance optimizations
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA cache_size=10000")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        
        await self._create_schema()
        await self._create_indexes()
        
        logger.info("✅ Enhanced resource database initialized with optimized schema")
    
    async def _create_schema(self):
        """Create comprehensive resource database schema"""
        # Main resources table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS resources (
                id TEXT PRIMARY KEY,
                embedding_id TEXT,
                message_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                url TEXT,
                title TEXT,
                description TEXT,
                quality_score REAL NOT NULL DEFAULT 0.0,
                quality_level TEXT NOT NULL DEFAULT 'fair',
                validation_status TEXT NOT NULL DEFAULT 'pending',
                validation_timestamp DATETIME,
                content_hash TEXT,
                domain TEXT,
                file_size INTEGER,
                content_type TEXT,
                language TEXT,
                author TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed DATETIME,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        # Resource metadata table for flexible key-value storage
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS resource_metadata (
                resource_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                value_type TEXT DEFAULT 'string',
                PRIMARY KEY (resource_id, key),
                FOREIGN KEY (resource_id) REFERENCES resources (id) ON DELETE CASCADE
            )
        """)
        
        # Resource validation history
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS resource_validation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resource_id TEXT NOT NULL,
                validation_status TEXT NOT NULL,
                response_time REAL,
                http_status INTEGER,
                content_length INTEGER,
                error_message TEXT,
                validated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (resource_id) REFERENCES resources (id) ON DELETE CASCADE
            )
        """)
        
        # Resource relationships (for clustering and deduplication)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS resource_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_resource_id TEXT NOT NULL,
                target_resource_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                confidence_score REAL DEFAULT 0.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_resource_id) REFERENCES resources (id) ON DELETE CASCADE,
                FOREIGN KEY (target_resource_id) REFERENCES resources (id) ON DELETE CASCADE
            )
        """)
        
        # Resource usage analytics
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS resource_usage_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resource_id TEXT NOT NULL,
                user_id TEXT,
                access_type TEXT NOT NULL,
                query_context TEXT,
                result_rank INTEGER,
                clicked BOOLEAN DEFAULT FALSE,
                accessed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (resource_id) REFERENCES resources (id) ON DELETE CASCADE
            )
        """)
        
        # Resource quality scores over time
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS resource_quality_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resource_id TEXT NOT NULL,
                quality_score REAL NOT NULL,
                quality_factors TEXT,  -- JSON
                calculated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (resource_id) REFERENCES resources (id) ON DELETE CASCADE
            )
        """)
        
        self.connection.commit()
    
    async def _create_indexes(self):
        """Create optimized indexes for query performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_resources_message_id ON resources(message_id)",
            "CREATE INDEX IF NOT EXISTS idx_resources_channel_id ON resources(channel_id)",
            "CREATE INDEX IF NOT EXISTS idx_resources_type ON resources(resource_type)",
            "CREATE INDEX IF NOT EXISTS idx_resources_quality_score ON resources(quality_score)",
            "CREATE INDEX IF NOT EXISTS idx_resources_validation_status ON resources(validation_status)",
            "CREATE INDEX IF NOT EXISTS idx_resources_domain ON resources(domain)",
            "CREATE INDEX IF NOT EXISTS idx_resources_created_at ON resources(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_resources_active ON resources(is_active)",
            "CREATE INDEX IF NOT EXISTS idx_resources_compound ON resources(resource_type, quality_score, is_active)",
            
            "CREATE INDEX IF NOT EXISTS idx_validation_history_resource_id ON resource_validation_history(resource_id)",
            "CREATE INDEX IF NOT EXISTS idx_validation_history_timestamp ON resource_validation_history(validated_at)",
            
            "CREATE INDEX IF NOT EXISTS idx_relationships_source ON resource_relationships(source_resource_id)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_target ON resource_relationships(target_resource_id)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_type ON resource_relationships(relationship_type)",
            
            "CREATE INDEX IF NOT EXISTS idx_usage_analytics_resource_id ON resource_usage_analytics(resource_id)",
            "CREATE INDEX IF NOT EXISTS idx_usage_analytics_accessed_at ON resource_usage_analytics(accessed_at)",
            
            "CREATE INDEX IF NOT EXISTS idx_quality_history_resource_id ON resource_quality_history(resource_id)",
            "CREATE INDEX IF NOT EXISTS idx_quality_history_timestamp ON resource_quality_history(calculated_at)"
        ]
        
        for index_sql in indexes:
            self.connection.execute(index_sql)
        
        self.connection.commit()
        logger.info("✅ Database indexes created for optimal query performance")
    
    async def store_resource(self, resource: ResourceMetadata, validation: ValidationResult, embedding_id: Optional[str] = None) -> bool:
        """Store a resource with validation results and vector integration"""
        try:
            # Create content hash for deduplication
            content_hash = self._calculate_content_hash(resource)
            
            # Check for existing resource
            existing = await self.get_resource_by_content_hash(content_hash)
            if existing:
                # Update existing resource
                return await self.update_resource(existing['id'], resource, validation)
            
            # Prepare resource data
            resource_data = {
                'id': resource.id,
                'embedding_id': embedding_id,
                'message_id': resource.message_id,
                'channel_id': resource.channel_id,
                'resource_type': resource.type.value,
                'url': resource.url,
                'title': resource.title,
                'description': resource.description,
                'quality_score': resource.quality_score,
                'quality_level': resource.quality_level.value,
                'validation_status': validation.status.value,
                'validation_timestamp': validation.validation_timestamp.isoformat() if validation.validation_timestamp else None,
                'content_hash': content_hash,
                'domain': resource.domain,
                'file_size': resource.file_size,
                'content_type': resource.content_type,
                'language': resource.language,
                'author': resource.author,
                'created_at': resource.created_at.isoformat() if resource.created_at else datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # Insert resource
            placeholders = ', '.join(['?' for _ in resource_data])
            columns = ', '.join(resource_data.keys())
            values = list(resource_data.values())
            
            self.connection.execute(
                f"INSERT INTO resources ({columns}) VALUES ({placeholders})",
                values
            )
            
            # Store metadata (keywords, topics, etc.)
            await self._store_resource_metadata(resource.id, resource)
            
            # Store validation history
            await self._store_validation_result(resource.id, validation)
            
            # Store initial quality score
            await self._store_quality_score(resource.id, resource.quality_score)
            
            self.connection.commit()
            
            logger.debug(f"✅ Stored resource {resource.id} with validation status {validation.status.value}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing resource {resource.id}: {e}")
            self.connection.rollback()
            return False
    
    async def update_resource(self, resource_id: str, resource: ResourceMetadata, validation: ValidationResult) -> bool:
        """Update existing resource with new validation data"""
        try:
            update_data = {
                'quality_score': resource.quality_score,
                'quality_level': resource.quality_level.value,
                'validation_status': validation.status.value,
                'validation_timestamp': validation.validation_timestamp.isoformat() if validation.validation_timestamp else None,
                'updated_at': datetime.now().isoformat()
            }
            
            set_clause = ', '.join([f"{key} = ?" for key in update_data.keys()])
            values = list(update_data.values()) + [resource_id]
            
            self.connection.execute(
                f"UPDATE resources SET {set_clause} WHERE id = ?",
                values
            )
            
            # Update validation history
            await self._store_validation_result(resource_id, validation)
            
            # Update quality score history
            await self._store_quality_score(resource_id, resource.quality_score)
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating resource {resource_id}: {e}")
            self.connection.rollback()
            return False
    
    async def get_resources_by_type(self, resource_type: ResourceType, limit: int = 100, quality_threshold: float = 0.5) -> List[Dict]:
        """Get resources by type with quality filtering"""
        cursor = self.connection.execute("""
            SELECT * FROM resources 
            WHERE resource_type = ? 
              AND quality_score >= ? 
              AND is_active = TRUE 
            ORDER BY quality_score DESC, created_at DESC 
            LIMIT ?
        """, (resource_type.value, quality_threshold, limit))
        
        return [dict(row) for row in cursor.fetchall()]
    
    async def get_resources_by_channel(self, channel_id: str, limit: int = 100) -> List[Dict]:
        """Get resources from specific channel"""
        cursor = self.connection.execute("""
            SELECT * FROM resources 
            WHERE channel_id = ? 
              AND is_active = TRUE 
            ORDER BY quality_score DESC, created_at DESC 
            LIMIT ?
        """, (channel_id, limit))
        
        return [dict(row) for row in cursor.fetchall()]
    
    async def get_high_quality_resources(self, limit: int = 50, min_quality: float = 0.8) -> List[Dict]:
        """Get highest quality resources across all types"""
        cursor = self.connection.execute("""
            SELECT r.*, COUNT(ua.id) as usage_count
            FROM resources r
            LEFT JOIN resource_usage_analytics ua ON r.id = ua.resource_id
            WHERE r.quality_score >= ? 
              AND r.is_active = TRUE 
              AND r.validation_status IN ('healthy', 'degraded')
            GROUP BY r.id
            ORDER BY r.quality_score DESC, usage_count DESC, r.created_at DESC 
            LIMIT ?
        """, (min_quality, limit))
        
        return [dict(row) for row in cursor.fetchall()]
    
    async def get_resource_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive resource analytics"""
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Basic statistics
        cursor = self.connection.execute("""
            SELECT 
                COUNT(*) as total_resources,
                AVG(quality_score) as avg_quality,
                COUNT(CASE WHEN validation_status = 'healthy' THEN 1 END) as healthy_resources,
                COUNT(CASE WHEN validation_status = 'broken' THEN 1 END) as broken_resources,
                COUNT(CASE WHEN created_at >= ? THEN 1 END) as recent_resources
            FROM resources 
            WHERE is_active = TRUE
        """, (since_date,))
        
        basic_stats = dict(cursor.fetchone())
        
        # Resource type distribution
        cursor = self.connection.execute("""
            SELECT resource_type, COUNT(*) as count, AVG(quality_score) as avg_quality
            FROM resources 
            WHERE is_active = TRUE
            GROUP BY resource_type
            ORDER BY count DESC
        """)
        
        type_distribution = [dict(row) for row in cursor.fetchall()]
        
        # Domain analysis
        cursor = self.connection.execute("""
            SELECT domain, COUNT(*) as count, AVG(quality_score) as avg_quality
            FROM resources 
            WHERE domain IS NOT NULL AND is_active = TRUE
            GROUP BY domain
            ORDER BY count DESC
            LIMIT 20
        """)
        
        domain_stats = [dict(row) for row in cursor.fetchall()]
        
        # Quality trends
        cursor = self.connection.execute("""
            SELECT 
                DATE(calculated_at) as date,
                AVG(quality_score) as avg_quality,
                COUNT(*) as measurements
            FROM resource_quality_history
            WHERE calculated_at >= ?
            GROUP BY DATE(calculated_at)
            ORDER BY date DESC
        """, (since_date,))
        
        quality_trends = [dict(row) for row in cursor.fetchall()]
        
        return {
            'basic_statistics': basic_stats,
            'type_distribution': type_distribution,
            'domain_statistics': domain_stats,
            'quality_trends': quality_trends,
            'generated_at': datetime.now().isoformat()
        }
    
    async def track_resource_access(self, resource_id: str, user_id: Optional[str] = None, access_type: str = "view", query_context: Optional[str] = None):
        """Track resource access for analytics"""
        try:
            # Update access count in main table
            self.connection.execute("""
                UPDATE resources 
                SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP 
                WHERE id = ?
            """, (resource_id,))
            
            # Record detailed access analytics
            self.connection.execute("""
                INSERT INTO resource_usage_analytics 
                (resource_id, user_id, access_type, query_context, accessed_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (resource_id, user_id, access_type, query_context))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"❌ Error tracking resource access: {e}")
    
    async def _store_resource_metadata(self, resource_id: str, resource: ResourceMetadata):
        """Store resource metadata in flexible key-value format"""
        metadata_items = []
        
        # Store keywords
        if resource.keywords:
            metadata_items.append((resource_id, 'keywords', json.dumps(resource.keywords), 'json'))
        
        # Store topics
        if resource.topics:
            metadata_items.append((resource_id, 'topics', json.dumps(resource.topics), 'json'))
        
        # Store additional metadata
        if hasattr(resource, 'relevance_score'):
            metadata_items.append((resource_id, 'relevance_score', str(resource.relevance_score), 'float'))
        
        for item in metadata_items:
            self.connection.execute("""
                INSERT OR REPLACE INTO resource_metadata 
                (resource_id, key, value, value_type) VALUES (?, ?, ?, ?)
            """, item)
    
    async def _store_validation_result(self, resource_id: str, validation: ValidationResult):
        """Store validation result in history"""
        self.connection.execute("""
            INSERT INTO resource_validation_history 
            (resource_id, validation_status, response_time, http_status, content_length, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            resource_id,
            validation.status.value,
            validation.response_time,
            validation.http_status,
            validation.content_length,
            validation.error_message
        ))
    
    async def _store_quality_score(self, resource_id: str, quality_score: float, factors: Optional[Dict] = None):
        """Store quality score in history for trend analysis"""
        self.connection.execute("""
            INSERT INTO resource_quality_history 
            (resource_id, quality_score, quality_factors)
            VALUES (?, ?, ?)
        """, (resource_id, quality_score, json.dumps(factors) if factors else None))
    
    def _calculate_content_hash(self, resource: ResourceMetadata) -> str:
        """Calculate content hash for deduplication"""
        content_parts = [
            resource.url or '',
            resource.title or '',
            resource.description or '',
            resource.type.value
        ]
        content_string = '|'.join(content_parts)
        return hashlib.sha256(content_string.encode()).hexdigest()
    
    async def get_resource_by_content_hash(self, content_hash: str) -> Optional[Dict]:
        """Get resource by content hash for deduplication"""
        cursor = self.connection.execute("""
            SELECT * FROM resources WHERE content_hash = ? AND is_active = TRUE
        """, (content_hash,))
        
        row = cursor.fetchone()
        return dict(row) if row else None
    
    async def cleanup_old_resources(self, days: int = 90):
        """Clean up old resources based on age and quality"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get resources to clean up
        cursor = self.connection.execute("""
            SELECT id FROM resources 
            WHERE created_at < ? AND quality_score < 0.3
        """, (cutoff_date,))
        
        old_resources = cursor.fetchall()
        
        if old_resources:
            resource_ids = [row[0] for row in old_resources]
            
            # Delete related data first
            self.connection.execute("DELETE FROM resource_metadata WHERE resource_id IN ({})".format(
                ','.join(['?' for _ in resource_ids])), resource_ids)
            self.connection.execute("DELETE FROM resource_validation_history WHERE resource_id IN ({})".format(
                ','.join(['?' for _ in resource_ids])), resource_ids)
            self.connection.execute("DELETE FROM resource_relationships WHERE source_resource_id IN ({}) OR target_resource_id IN ({})".format(
                ','.join(['?' for _ in resource_ids]), ','.join(['?' for _ in resource_ids])), resource_ids + resource_ids)
            self.connection.execute("DELETE FROM resource_usage_analytics WHERE resource_id IN ({})".format(
                ','.join(['?' for _ in resource_ids])), resource_ids)
            self.connection.execute("DELETE FROM resource_quality_history WHERE resource_id IN ({})".format(
                ','.join(['?' for _ in resource_ids])), resource_ids)
            
            # Delete main resources
            self.connection.execute("DELETE FROM resources WHERE id IN ({})".format(
                ','.join(['?' for _ in resource_ids])), resource_ids)
            
            self.connection.commit()
            logger.info(f"🧹 Cleaned up {len(old_resources)} old resources")
    
    async def clear_all_resources(self):
        """Clear all resources from the database (for complete reset)"""
        logger.info("🗑️ Clearing all resources from database")
        
        # Delete all data from all tables
        tables = [
            'resource_usage_analytics',
            'resource_quality_history', 
            'resource_relationships',
            'resource_validation_history',
            'resource_metadata',
            'resources'
        ]
        
        for table in tables:
            self.connection.execute(f"DELETE FROM {table}")
        
        self.connection.commit()
        logger.info("✅ All resources cleared from database")

    async def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("🔒 Enhanced resource database connection closed") 