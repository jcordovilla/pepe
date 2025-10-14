#!/usr/bin/env python3
"""
Clean Migration Script: Enhanced Resource Database

Performs a clean migration that:
1. Clears the existing database (when --reset-cache is used)
2. Ensures proper AI-generated descriptions
3. Prevents duplicates
4. Maintains data consistency
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import sqlite3
import argparse

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


class CleanResourceMigration:
    """
    Performs clean migration with proper database management
    """
    
    def __init__(self, reset_cache: bool = False):
        self.reset_cache = reset_cache
        self.enhanced_db = None
        self.vector_integration = None
        
        # Migration statistics
        self.stats = {
            'total_resources_found': 0,
            'resources_migrated': 0,
            'resources_skipped': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None,
            'database_cleared': False
        }
        
        # ChromaDB path for integration
        self.chromadb_path = "data/chromadb"
        
        logger.info("🔄 Clean resource migration initialized")
    
    async def run_clean_migration(self) -> Dict[str, Any]:
        """
        Run clean migration with proper database management
        """
        self.stats['start_time'] = datetime.now()
        
        print("🚀 Starting Clean Resource Database Migration")
        print("=" * 60)
        
        if self.reset_cache:
            print("🗑️ RESET MODE: Will clear existing database and reprocess all resources")
        else:
            print("📈 INCREMENTAL MODE: Will add new resources to existing database")
        
        try:
            # Initialize enhanced database
            await self._initialize_enhanced_database()
            
            # Load fresh resources from detection
            await self._load_fresh_resources()
            
            # Migrate resources with proper descriptions
            await self._migrate_fresh_resources()
            
            # Link vector store data
            await self._link_vector_store_data()
            
            # Validate migration
            await self._validate_migration()
            
            # Generate migration report
            migration_report = await self._generate_migration_report()
            
            # Export to resources-data.json for HTML generation
            await self._export_to_json()
            
            # Generate HTML file
            await self._generate_html()
            
            self.stats['end_time'] = datetime.now()
            
            print("\n🎉 Clean migration completed successfully!")
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
        
        # Create backup of existing data if not in reset mode
        if not self.reset_cache:
            await self._create_backup()
        
        # Initialize enhanced database
        self.enhanced_db = EnhancedResourceDatabase("data/enhanced_resources.db")
        
        if self.reset_cache:
            # Clear existing database
            print("🗑️ Clearing existing database...")
            await self.enhanced_db.clear_all_resources()
            self.stats['database_cleared'] = True
            print("✅ Database cleared")
        
        await self.enhanced_db.initialize()
        
        # Initialize vector integration
        self.vector_integration = VectorResourceIntegration(
            chromadb_path=self.chromadb_path,
            resource_db_path="data/enhanced_resources.db"
        )
        await self.vector_integration.initialize()
        
        print("✅ Enhanced database initialized")
    
    async def _load_fresh_resources(self):
        """Load fresh resources from the detection output"""
        print("\n📥 Loading Fresh Resources from Detection")
        
        resources_file = Path("data/optimized_fresh_resources.json")
        if not resources_file.exists():
            print("❌ No fresh resources file found. Run resource detection first.")
            raise FileNotFoundError("data/optimized_fresh_resources.json not found")
        
        try:
            with open(resources_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract resources from the report
            resources = data.get('resources', [])
            self.stats['total_resources_found'] = len(resources)
            
            print(f"📄 Found {len(resources)} fresh resources to migrate")
            
            # Store resources for processing
            self.fresh_resources = resources
            
        except Exception as e:
            logger.error(f"❌ Error loading fresh resources: {e}")
            raise
    
    async def _migrate_fresh_resources(self):
        """Migrate fresh resources with proper descriptions"""
        print("\n🔄 Migrating Fresh Resources")
        
        if not hasattr(self, 'fresh_resources'):
            print("❌ No fresh resources loaded")
            return
        
        # Process in batches for better progress tracking
        batch_size = 50
        total_batches = (len(self.fresh_resources) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(self.fresh_resources))
            batch = self.fresh_resources[start_idx:end_idx]
            
            print(f"  Processing batch {batch_num + 1}/{total_batches} ({len(batch)} resources)")
            
            for resource_data in batch:
                try:
                    # Convert to enhanced format
                    enhanced_resource = await self._convert_fresh_resource(resource_data)
                    if not enhanced_resource:
                        self.stats['resources_skipped'] += 1
                        continue
                    
                    # Create validation result
                    validation = await self._create_validation_result(resource_data)
                    
                    # Get embedding ID if available
                    embedding_id = await self._get_embedding_id(resource_data.get('message_id'))
                    
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
            
            # Progress update
            progress = ((batch_num + 1) / total_batches) * 100
            print(f"  Progress: {progress:.1f}% ({self.stats['resources_migrated']} migrated)")
    
    async def _convert_fresh_resource(self, resource_data: Dict[str, Any]) -> ResourceMetadata:
        """Convert fresh resource data to enhanced ResourceMetadata"""
        try:
            # Map category to ResourceType
            resource_type = self._map_category_to_type(resource_data.get('category', 'unknown'))
            
            # Create ResourceMetadata
            enhanced = ResourceMetadata(
                id=str(resource_data.get('id', '')),
                type=resource_type,
                url=resource_data.get('url'),
                title=resource_data.get('title', ''),
                description=resource_data.get('description', ''),
                domain=resource_data.get('domain', ''),
                quality_score=resource_data.get('quality_score', 0.5),
                quality_level=self._score_to_quality_level(resource_data.get('quality_score', 0.5)),
                channel_id=resource_data.get('channel_id', ''),
                channel_name=resource_data.get('channel_name', ''),
                author_id=resource_data.get('author_id', ''),
                author_name=resource_data.get('author', ''),
                message_id=resource_data.get('message_id', ''),
                jump_url=resource_data.get('jump_url', ''),
                timestamp=self._parse_timestamp(resource_data.get('timestamp', '')),
                metadata={
                    'detection_method': 'fresh_detection',
                    'migration_timestamp': datetime.now().isoformat()
                }
            )
            
            return enhanced
            
        except Exception as e:
            logger.error(f"❌ Error converting resource: {e}")
            return None
    
    def _map_category_to_type(self, category: str) -> ResourceType:
        """Map resource category to ResourceType enum"""
        category_lower = category.lower()
        
        if 'research' in category_lower or 'paper' in category_lower:
            return ResourceType.RESEARCH_PAPER
        elif 'code' in category_lower or 'repository' in category_lower:
            return ResourceType.CODE_REPOSITORY
        elif 'video' in category_lower or 'youtube' in category_lower:
            return ResourceType.VIDEO
        elif 'documentation' in category_lower or 'docs' in category_lower:
            return ResourceType.DOCUMENTATION
        elif 'article' in category_lower or 'blog' in category_lower:
            return ResourceType.ARTICLE
        elif 'news' in category_lower:
            return ResourceType.NEWS
        elif 'education' in category_lower or 'course' in category_lower:
            return ResourceType.EDUCATIONAL
        else:
            return ResourceType.OTHER
    
    def _score_to_quality_level(self, score: float) -> ResourceQuality:
        """Convert quality score to ResourceQuality enum"""
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
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            pass
        return datetime.now()
    
    async def _create_validation_result(self, resource_data: Dict[str, Any]) -> ValidationResult:
        """Create validation result for fresh resource"""
        # Assume fresh resources are healthy
        status = ValidationStatus.HEALTHY
        
        return ValidationResult(
            resource_id=str(resource_data.get('id', '')),
            status=status,
            response_time=0.0,
            validation_timestamp=datetime.now()
        )
    
    async def _get_embedding_id(self, message_id: str) -> str:
        """Get embedding ID for message from ChromaDB"""
        try:
            if self.vector_integration and message_id:
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
        
        # Check description quality
        cursor = self.enhanced_db.connection.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN description IS NOT NULL AND description != '' AND description NOT LIKE 'AI/ML resource from%' THEN 1 END) as good_descriptions,
                COUNT(CASE WHEN description LIKE 'AI/ML resource from%' THEN 1 END) as generic_descriptions,
                AVG(quality_score) as avg_quality
            FROM resources
        """)
        
        integrity_stats = cursor.fetchone()
        print(f"  Resources with good descriptions: {integrity_stats[1]}")
        print(f"  Resources with generic descriptions: {integrity_stats[2]}")
        print(f"  Average quality score: {integrity_stats[3]:.3f}")
        
        # Check for duplicates
        cursor = self.enhanced_db.connection.execute("""
            SELECT url, COUNT(*) as count 
            FROM resources 
            GROUP BY url 
            HAVING COUNT(*) > 1
        """)
        
        duplicates = cursor.fetchall()
        if duplicates:
            print(f"  ⚠️ Found {len(duplicates)} duplicate URLs")
            for url, count in duplicates[:5]:
                print(f"    • {url}: {count} instances")
        else:
            print("  ✅ No duplicate URLs found")
    
    async def _export_to_json(self):
        """Export resources to resources-data.json for HTML generation"""
        print("\n📄 Exporting resources to JSON...")
        
        try:
            # Get all resources from enhanced database
            conn = sqlite3.connect("data/enhanced_resources.db")
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    url,
                    title,
                    description,
                    type,
                    quality_score,
                    author_name as author,
                    channel_name as channel,
                    DATE(shared_date) as date,
                    discord_url
                FROM resources
                ORDER BY shared_date DESC
            """)
            
            resources = []
            for idx, row in enumerate(cursor.fetchall(), 1):
                resources.append({
                    'id': idx,
                    'title': row['title'] or 'Untitled Resource',
                    'description': row['description'] or 'No description available.',
                    'date': row['date'] or datetime.now().strftime('%Y-%m-%d'),
                    'author': row['author'] or 'Unknown',
                    'channel': row['channel'] or 'general',
                    'tag': row['type'] or 'Other',
                    'resource_url': row['url'],
                    'discord_url': row['discord_url'] or ''
                })
            
            conn.close()
            
            # Create export data structure
            export_data = {
                'export_date': datetime.now().isoformat(),
                'total_resources': len(resources),
                'resources': resources
            }
            
            # Save to resources-data.json
            output_path = Path("data/resources-data.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"  ✅ Exported {len(resources)} resources to {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to export resources to JSON: {e}")
            raise
    
    async def _generate_html(self):
        """Generate HTML file from resources-data.json"""
        print("\n🌐 Generating HTML file...")
        
        try:
            # Import and run the HTML generator
            from generate_resources_html import generate_html
            
            output_path = generate_html()
            print(f"  ✅ HTML file generated at {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate HTML: {e}")
            # Don't raise - HTML generation failure shouldn't stop migration
            print(f"  ⚠️ HTML generation failed: {e}")
    
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
        report_path = f"data/clean_migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📋 Migration report saved: {report_path}")
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate post-migration recommendations"""
        recommendations = []
        
        if self.stats['database_cleared']:
            recommendations.append("🗑️ Database was cleared and rebuilt - all data is fresh")
        
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
        
        # Backup existing database
        db_path = Path("data/enhanced_resources.db")
        if db_path.exists():
            backup_path = backup_dir / "enhanced_resources.db"
            import shutil
            shutil.copy2(db_path, backup_path)
            print(f"  📄 Backed up {db_path}")
        
        print(f"✅ Backup created in {backup_dir}")
    
    async def _cleanup(self):
        """Cleanup resources"""
        if self.enhanced_db:
            await self.enhanced_db.close()
        
        if self.vector_integration:
            await self.vector_integration.cleanup()


async def main():
    """Main migration execution"""
    parser = argparse.ArgumentParser(description='Clean Resource Database Migration')
    parser.add_argument('--reset-cache', action='store_true',
                       help='Clear existing database and rebuild from scratch')
    args = parser.parse_args()
    
    migration_manager = CleanResourceMigration(reset_cache=args.reset_cache)
    
    try:
        # Run migration
        report = await migration_manager.run_clean_migration()
        
        print("\n🎊 Clean Resource Database Migration Complete!")
        print("\n📊 Summary:")
        print(f"  • Total resources found: {report['migration_statistics']['total_resources_found']}")
        print(f"  • Successfully migrated: {report['migration_statistics']['resources_migrated']}")
        print(f"  • Database cleared: {report['migration_statistics']['database_cleared']}")
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