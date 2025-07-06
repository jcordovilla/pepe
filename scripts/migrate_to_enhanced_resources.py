#!/usr/bin/env python3
"""
Migration Script: Enhanced Resource Database

Migrates existing resource data to the new enhanced resource database
with vector store integration and comprehensive tracking.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import sqlite3

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.database.enhanced_resource_database import EnhancedResourceDatabase
from agentic.database.vector_resource_integration import VectorResourceIntegration
from agentic.services.enhanced_resource_detector import ResourceMetadata, ResourceType, ResourceQuality
from agentic.services.resource_validation_pipeline import ValidationResult, ValidationStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResourceMigrationManager:
    """
    Manages migration from existing resource storage to enhanced database
    """
    
    def __init__(self):
        self.enhanced_db = None
        self.vector_integration = None
        
        # Migration statistics
        self.stats = {
            'total_resources_found': 0,
            'resources_migrated': 0,
            'resources_skipped': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Data sources
        self.legacy_resource_files = [
            "data/detected_resources/detected_resources_20250602_225804.json"
        ]
        
        # ChromaDB path for integration
        self.chromadb_path = "data/chromadb"
        
        logger.info("🔄 Resource migration manager initialized")
    
    async def run_migration(self) -> Dict[str, Any]:
        """
        Run complete migration from legacy to enhanced resource system
        """
        self.stats['start_time'] = datetime.now()
        
        print("🚀 Starting Enhanced Resource Database Migration")
        print("=" * 60)
        
        try:
            # Initialize enhanced database
            await self._initialize_enhanced_database()
            
            # Migrate legacy resource files
            await self._migrate_legacy_resources()
            
            # Migrate vector store links
            await self._link_vector_store_data()
            
            # Validate migration
            await self._validate_migration()
            
            # Generate migration report
            migration_report = await self._generate_migration_report()
            
            self.stats['end_time'] = datetime.now()
            
            print("\n🎉 Migration completed successfully!")
            print(f"📊 Migrated {self.stats['resources_migrated']}/{self.stats['total_resources_found']} resources")
            
            return migration_report
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            self.stats['end_time'] = datetime.now()
            raise
        
        finally:
            await self._cleanup()
    
    async def _initialize_enhanced_database(self):
        """Initialize the enhanced resource database"""
        print("\n📊 Initializing Enhanced Resource Database")
        
        # Create backup of existing data
        await self._create_backup()
        
        # Initialize enhanced database
        self.enhanced_db = EnhancedResourceDatabase("data/enhanced_resources.db")
        await self.enhanced_db.initialize()
        
        # Initialize vector integration
        self.vector_integration = VectorResourceIntegration(
            chromadb_path=self.chromadb_path,
            resource_db_path="data/enhanced_resources.db"
        )
        await self.vector_integration.initialize()
        
        print("✅ Enhanced database initialized")
    
    async def _migrate_legacy_resources(self):
        """Migrate resources from legacy JSON files"""
        print("\n📥 Migrating Legacy Resource Files")
        
        for file_path in self.legacy_resource_files:
            if not Path(file_path).exists():
                logger.warning(f"⚠️ Legacy file not found: {file_path}")
                continue
            
            print(f"📄 Processing: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    legacy_resources = json.load(f)
                
                if not isinstance(legacy_resources, list):
                    logger.warning(f"⚠️ Unexpected format in {file_path}")
                    continue
                
                print(f"  Found {len(legacy_resources)} legacy resources")
                self.stats['total_resources_found'] += len(legacy_resources)
                
                # Process in batches
                batch_size = 100
                for i in range(0, len(legacy_resources), batch_size):
                    batch = legacy_resources[i:i + batch_size]
                    await self._process_resource_batch(batch)
                    
                    progress = ((i + len(batch)) / len(legacy_resources)) * 100
                    print(f"  Progress: {progress:.1f}% ({i + len(batch)}/{len(legacy_resources)})")
                
                print(f"  ✅ Completed migration of {file_path}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {file_path}: {e}")
                self.stats['errors'] += 1
    
    async def _process_resource_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of legacy resources"""
        for legacy_resource in batch:
            try:
                # Convert legacy resource to enhanced format
                enhanced_resource = await self._convert_legacy_resource(legacy_resource)
                if not enhanced_resource:
                    self.stats['resources_skipped'] += 1
                    continue
                
                # Create validation result
                validation = await self._create_validation_result(legacy_resource)
                
                # Get embedding ID if available
                embedding_id = await self._get_embedding_id(legacy_resource.get('message_id'))
                
                # Store in enhanced database
                success = await self.enhanced_db.store_resource(enhanced_resource, validation, embedding_id)
                
                if success:
                    self.stats['resources_migrated'] += 1
                else:
                    self.stats['resources_skipped'] += 1
                    logger.warning(f"⚠️ Failed to store resource: {enhanced_resource.id}")
                
            except Exception as e:
                logger.error(f"❌ Error processing resource: {e}")
                self.stats['errors'] += 1
    
    async def _convert_legacy_resource(self, legacy: Dict[str, Any]) -> ResourceMetadata:
        """Convert legacy resource format to enhanced ResourceMetadata"""
        try:
            # Map legacy type to new ResourceType
            resource_type = self._map_legacy_type(legacy.get('type', 'unknown'))
            
            # Create ResourceMetadata
            enhanced = ResourceMetadata(
                id=str(legacy.get('id', '')),
                type=resource_type,
                url=legacy.get('url'),
                domain=legacy.get('domain'),
                title=self._extract_title(legacy),
                description=legacy.get('content_preview', ''),
                quality_score=self._calculate_legacy_quality_score(legacy),
                quality_level=ResourceQuality.FAIR,  # Default, will be recalculated
                message_id=legacy.get('message_id'),
                channel_id=legacy.get('channel_id'),
                author=legacy.get('author'),
                created_at=self._parse_timestamp(legacy.get('timestamp')),
                file_size=legacy.get('size'),
                content_type=legacy.get('content_type')
            )
            
            # Set quality level based on score
            enhanced.quality_level = self._score_to_quality_level(enhanced.quality_score)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"❌ Error converting legacy resource: {e}")
            return None
    
    def _map_legacy_type(self, legacy_type: str) -> ResourceType:
        """Map legacy resource type to new ResourceType enum"""
        type_mapping = {
            'url': ResourceType.UNKNOWN,
            'attachment': ResourceType.PDF_DOCUMENT,
            'code': ResourceType.CODE_SNIPPET,
            'document': ResourceType.PDF_DOCUMENT,
            'video': ResourceType.VIDEO_TUTORIAL,
            'image': ResourceType.IMAGE
        }
        
        return type_mapping.get(legacy_type, ResourceType.UNKNOWN)
    
    def _extract_title(self, legacy: Dict[str, Any]) -> str:
        """Extract title from legacy resource"""
        # Try various fields that might contain title information
        if legacy.get('filename'):
            return legacy['filename']
        
        if legacy.get('content_preview'):
            # Extract first line as title
            first_line = legacy['content_preview'].split('\n')[0].strip()
            if len(first_line) > 10 and len(first_line) < 100:
                return first_line
        
        if legacy.get('url'):
            # Extract filename from URL
            url_parts = legacy['url'].split('/')
            if url_parts:
                filename = url_parts[-1]
                if '.' in filename:
                    return filename
        
        return "Untitled Resource"
    
    def _calculate_legacy_quality_score(self, legacy: Dict[str, Any]) -> float:
        """Calculate quality score for legacy resource"""
        score = 0.5  # Base score
        
        # Domain-based scoring
        domain = legacy.get('domain', '').lower()
        high_quality_domains = [
            'arxiv.org', 'github.com', 'openai.com', 'deeplearning.ai',
            'tensorflow.org', 'pytorch.org', 'huggingface.co'
        ]
        
        if any(hq_domain in domain for hq_domain in high_quality_domains):
            score += 0.3
        
        # File size considerations for attachments
        if legacy.get('size'):
            size = legacy['size']
            if size > 100000:  # 100KB+
                score += 0.1
            if size > 1000000:  # 1MB+
                score += 0.1
        
        # Content preview quality
        if legacy.get('content_preview'):
            content = legacy['content_preview']
            if len(content) > 100:
                score += 0.1
            if any(keyword in content.lower() for keyword in ['research', 'paper', 'tutorial', 'guide']):
                score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _score_to_quality_level(self, score: float) -> ResourceQuality:
        """Convert quality score to quality level"""
        if score >= 0.9:
            return ResourceQuality.EXCELLENT
        elif score >= 0.7:
            return ResourceQuality.HIGH
        elif score >= 0.5:
            return ResourceQuality.GOOD
        elif score >= 0.3:
            return ResourceQuality.FAIR
        else:
            return ResourceQuality.POOR
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime"""
        try:
            if timestamp_str:
                # Handle ISO format with timezone
                if '+' in timestamp_str:
                    timestamp_str = timestamp_str.split('+')[0]
                return datetime.fromisoformat(timestamp_str.replace('Z', ''))
            return datetime.now()
        except:
            return datetime.now()
    
    async def _create_validation_result(self, legacy: Dict[str, Any]) -> ValidationResult:
        """Create validation result for legacy resource"""
        # Assume legacy resources are generally healthy unless we know otherwise
        status = ValidationStatus.HEALTHY
        
        # Check for indicators of broken resources
        if legacy.get('domain') in ['cdn.discordapp.com']:
            status = ValidationStatus.DEGRADED  # Discord CDN links may expire
        
        return ValidationResult(
            resource_id=str(legacy.get('id', '')),
            status=status,
            response_time=0.0,
            validation_timestamp=datetime.now()
        )
    
    async def _get_embedding_id(self, message_id: str) -> str:
        """Get embedding ID for message from ChromaDB"""
        try:
            if self.vector_integration:
                return await self.vector_integration._get_embedding_id_for_message(message_id)
            return None
        except Exception as e:
            logger.debug(f"Could not get embedding ID for message {message_id}: {e}")
            return None
    
    async def _link_vector_store_data(self):
        """Link resources with vector store embeddings"""
        print("\n🔗 Linking with Vector Store Data")
        
        try:
            # Get all resources without embedding links
            cursor = self.enhanced_db.connection.execute("""
                SELECT id, message_id FROM resources 
                WHERE embedding_id IS NULL AND message_id IS NOT NULL
                LIMIT 1000
            """)
            
            unlinked_resources = cursor.fetchall()
            print(f"  Found {len(unlinked_resources)} resources to link")
            
            linked_count = 0
            for resource_id, message_id in unlinked_resources:
                embedding_id = await self._get_embedding_id(message_id)
                if embedding_id:
                    self.enhanced_db.connection.execute("""
                        UPDATE resources SET embedding_id = ? WHERE id = ?
                    """, (embedding_id, resource_id))
                    linked_count += 1
            
            self.enhanced_db.connection.commit()
            print(f"  ✅ Linked {linked_count} resources to vector embeddings")
            
        except Exception as e:
            logger.error(f"❌ Error linking vector store data: {e}")
    
    async def _validate_migration(self):
        """Validate the migration results"""
        print("\n🔍 Validating Migration Results")
        
        # Check resource counts
        cursor = self.enhanced_db.connection.execute("SELECT COUNT(*) FROM resources")
        db_count = cursor.fetchone()[0]
        
        print(f"  Database contains {db_count} resources")
        print(f"  Expected approximately {self.stats['total_resources_found']} resources")
        
        # Check data integrity
        cursor = self.enhanced_db.connection.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN url IS NOT NULL THEN 1 END) as with_urls,
                COUNT(CASE WHEN embedding_id IS NOT NULL THEN 1 END) as with_embeddings,
                AVG(quality_score) as avg_quality
            FROM resources
        """)
        
        integrity_stats = cursor.fetchone()
        print(f"  Resources with URLs: {integrity_stats[1]}")
        print(f"  Resources with embeddings: {integrity_stats[2]}")
        print(f"  Average quality score: {integrity_stats[3]:.3f}")
        
        # Test search functionality
        try:
            sample_resources = await self.vector_integration.search_resources_with_context(
                "machine learning", limit=5
            )
            print(f"  Test search returned {len(sample_resources)} resources")
            print("  ✅ Search functionality working")
        except Exception as e:
            logger.error(f"❌ Search test failed: {e}")
    
    async def _generate_migration_report(self) -> Dict[str, Any]:
        """Generate comprehensive migration report"""
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        # Get analytics from enhanced database
        analytics = await self.enhanced_db.get_resource_analytics(days=365)  # All time
        
        report = {
            'migration_timestamp': self.stats['end_time'].isoformat(),
            'duration_seconds': duration,
            'migration_statistics': self.stats,
            'resource_analytics': analytics,
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_path = f"data/migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📋 Migration report saved: {report_path}")
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate post-migration recommendations"""
        recommendations = []
        
        if self.stats['resources_migrated'] > 0:
            recommendations.append("✅ Migration completed successfully")
            recommendations.append("🔄 Consider running resource validation pipeline to update health status")
            recommendations.append("📊 Monitor resource quality scores and update thresholds as needed")
        
        if self.stats['resources_skipped'] > 0:
            recommendations.append(f"⚠️ {self.stats['resources_skipped']} resources were skipped - review logs for details")
        
        if self.stats['errors'] > 0:
            recommendations.append(f"❌ {self.stats['errors']} errors occurred - review error logs")
        
        recommendations.extend([
            "🔍 Set up automated resource validation schedules",
            "📈 Configure quality score monitoring and alerts",
            "🧹 Schedule periodic cleanup of low-quality resources",
            "📚 Update documentation with new resource API endpoints"
        ])
        
        return recommendations
    
    async def _create_backup(self):
        """Create backup of existing data"""
        print("💾 Creating backup of existing data")
        
        backup_dir = Path("data/backup") / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup existing resource files
        for file_path in self.legacy_resource_files:
            if Path(file_path).exists():
                backup_path = backup_dir / Path(file_path).name
                import shutil
                shutil.copy2(file_path, backup_path)
                print(f"  📄 Backed up {file_path}")
        
        print(f"✅ Backup created in {backup_dir}")
    
    async def _cleanup(self):
        """Cleanup resources"""
        if self.enhanced_db:
            await self.enhanced_db.close()
        
        if self.vector_integration:
            await self.vector_integration.cleanup()


async def main():
    """Main migration execution"""
    migration_manager = ResourceMigrationManager()
    
    try:
        # Run migration
        report = await migration_manager.run_migration()
        
        print("\n🎊 Enhanced Resource Database Migration Complete!")
        print("\n📊 Summary:")
        print(f"  • Total resources found: {report['migration_statistics']['total_resources_found']}")
        print(f"  • Successfully migrated: {report['migration_statistics']['resources_migrated']}")
        print(f"  • Duration: {report['duration_seconds']:.1f} seconds")
        
        if report['resource_analytics']['basic_statistics']:
            stats = report['resource_analytics']['basic_statistics']
            print(f"  • Average quality score: {stats['avg_quality']:.3f}")
            print(f"  • Healthy resources: {stats['healthy_resources']}")
        
        print("\n🚀 Next Steps:")
        for rec in report['recommendations']:
            print(f"  {rec}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main()) 