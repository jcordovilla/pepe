#!/usr/bin/env python3
"""
Resource Migration Script

This script migrates detected resources from the JSON file into the enhanced resources database.
It creates optimized indexes and generates additional metadata for fast searching.
"""

import json
import sqlite3
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResourceMigrator:
    def __init__(self, db_path: str = "data/resources.db"):
        self.db_path = db_path
        self.resources_file = "data/optimized_fresh_resources.json"
        
    def create_tables(self):
        """Create the enhanced resources database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create resources table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                domain TEXT NOT NULL,
                category TEXT NOT NULL,
                quality_score REAL NOT NULL,
                channel_name TEXT,
                author TEXT,
                timestamp TEXT,
                jump_url TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for fast searching
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_domain ON resources(domain)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON resources(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality_score ON resources(quality_score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_channel_name ON resources(channel_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_author ON resources(author)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON resources(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON resources(created_at)")
        
        # Create statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resource_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_resources INTEGER DEFAULT 0,
                categories_count TEXT,  -- JSON string
                domains_count TEXT,     -- JSON string
                channels_count TEXT,    -- JSON string
                quality_distribution TEXT, -- JSON string
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("✅ Database tables and indexes created")
    
    def load_resources(self) -> Dict[str, Any]:
        """Load resources from the JSON file."""
        if not Path(self.resources_file).exists():
            raise FileNotFoundError(f"Resources file not found: {self.resources_file}")
        
        with open(self.resources_file, 'r') as f:
            data = json.load(f)
        
        logger.info(f"📄 Loaded {len(data.get('resources', []))} resources from {self.resources_file}")
        return data
    
    def migrate_resources(self, reset_cache: bool = False):
        """Migrate resources to the enhanced database."""
        # Create tables
        self.create_tables()
        
        # Load resources
        data = self.load_resources()
        resources = data.get('resources', [])
        statistics = data.get('statistics', {})
        
        if not resources:
            logger.warning("⚠️ No resources to migrate")
            return
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing data if reset_cache is True
        if reset_cache:
            cursor.execute("DELETE FROM resources")
            cursor.execute("DELETE FROM resource_statistics")
            logger.info("🗑️ Cleared existing resources (reset mode)")
        
        # Insert resources
        inserted_count = 0
        skipped_count = 0
        
        for resource in resources:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO resources 
                    (url, domain, category, quality_score, channel_name, author, timestamp, jump_url, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    resource.get('url'),
                    resource.get('domain'),
                    resource.get('category'),
                    resource.get('quality_score', 0.0),
                    resource.get('channel_name'),
                    resource.get('author'),
                    resource.get('timestamp'),
                    resource.get('jump_url'),
                    resource.get('description')
                ))
                
                if cursor.rowcount > 0:
                    inserted_count += 1
                else:
                    skipped_count += 1
                    
            except Exception as e:
                logger.error(f"❌ Error inserting resource {resource.get('url', 'unknown')}: {e}")
                skipped_count += 1
        
        # Update statistics
        cursor.execute("""
            INSERT OR REPLACE INTO resource_statistics 
            (id, total_resources, categories_count, domains_count, channels_count, quality_distribution, last_updated)
            VALUES (1, ?, ?, ?, ?, ?, ?)
        """, (
            statistics.get('total_found', 0),
            json.dumps(statistics.get('categories', {})),
            json.dumps(statistics.get('domains', {})),
            json.dumps(statistics.get('channels', {})),
            json.dumps(statistics.get('quality_distribution', {})),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Migration completed:")
        logger.info(f"   📥 Inserted: {inserted_count} new resources")
        logger.info(f"   ⏭️ Skipped: {skipped_count} existing resources")
        logger.info(f"   📊 Total in database: {inserted_count + skipped_count}")
    
    def show_database_stats(self):
        """Show current database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM resources")
        total = cursor.fetchone()[0]
        
        # Get category distribution
        cursor.execute("""
            SELECT category, COUNT(*) as count 
            FROM resources 
            GROUP BY category 
            ORDER BY count DESC
        """)
        categories = cursor.fetchall()
        
        # Get domain distribution
        cursor.execute("""
            SELECT domain, COUNT(*) as count 
            FROM resources 
            GROUP BY domain 
            ORDER BY count DESC 
            LIMIT 10
        """)
        domains = cursor.fetchall()
        
        # Get quality distribution
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN quality_score >= 0.9 THEN 'excellent'
                    WHEN quality_score >= 0.8 THEN 'high'
                    WHEN quality_score >= 0.7 THEN 'good'
                    ELSE 'fair'
                END as quality_level,
                COUNT(*) as count
            FROM resources 
            GROUP BY quality_level 
            ORDER BY quality_score DESC
        """)
        quality_dist = cursor.fetchall()
        
        conn.close()
        
        print(f"\n📊 Enhanced Resources Database Statistics:")
        print(f"   📈 Total resources: {total}")
        
        if categories:
            print(f"\n📁 Categories:")
            for category, count in categories:
                print(f"   • {category}: {count}")
        
        if domains:
            print(f"\n🌐 Top Domains:")
            for domain, count in domains:
                print(f"   • {domain}: {count}")
        
        if quality_dist:
            print(f"\n⭐ Quality Distribution:")
            for quality, count in quality_dist:
                print(f"   • {quality}: {count}")

def main():
    parser = argparse.ArgumentParser(description="Migrate resources to enhanced database")
    parser.add_argument("--reset-cache", action="store_true", help="Clear existing resources before migration")
    args = parser.parse_args()
    
    try:
        migrator = ResourceMigrator()
        migrator.migrate_resources(reset_cache=args.reset_cache)
        migrator.show_database_stats()
        print("\n✅ Resource migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 