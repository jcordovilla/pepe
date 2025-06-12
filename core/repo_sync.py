#!/usr/bin/env python3
"""
Resource Synchronization Module

Syncs resources from the database to JSON files for use by FAISS index builders.
This replaces the archived legacy repo_sync with a simplified, focused approach.
"""

import os
import sys
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from urllib.parse import urlparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from db.db import SessionLocal, Resource

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sync_resources_to_json(output_path: str = "data/resources/detected_resources.json") -> int:
    """
    Sync all resources from database to JSON file.
    
    Args:
        output_path: Path to write JSON file
        
    Returns:
        Number of resources synced
    """
    logger.info("Syncing resources from database to JSON...")
    
    session = SessionLocal()
    try:
        # Query all resources from database
        resources = session.query(Resource).order_by(Resource.timestamp.desc()).all()
        logger.info(f"Found {len(resources)} resources in database")
        
        if not resources:
            logger.warning("No resources found in database")
            return 0
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        resource_dicts = []
        
        for res in resources:
            try:
                # Parse URL for domain
                domain = ""
                if res.url:
                    try:
                        parsed_url = urlparse(res.url)
                        domain = parsed_url.netloc.replace('www.', '')
                    except:
                        domain = "unknown"
                
                # Create resource dictionary
                resource_data = {
                    "id": res.id,
                    "title": res.name or res.url,
                    "description": res.description or res.context_snippet or "",
                    "date": res.timestamp.strftime("%Y-%m-%d") if res.timestamp else None,
                    "author": res.author_display or res.author,
                    "channel": res.channel_name or res.channel_id,
                    "tag": res.tag,
                    "resource_url": res.url,
                    "discord_url": res.jump_url,
                    "type": res.type,
                    "message_id": res.message_id,
                    "guild_id": res.guild_id,
                    "channel_id": res.channel_id,
                    "domain": domain
                }
                
                resource_dicts.append(resource_data)
                
            except Exception as e:
                logger.error(f"Failed to process resource {res.id}: {e}")
                continue
        
        # Write JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(resource_dicts, f, indent=2, default=str)
        
        logger.info(f"Successfully synced {len(resource_dicts)} resources to {output_path}")
        
        # Generate summary statistics
        tag_counts = {}
        domain_counts = {}
        for res in resource_dicts:
            tag = res.get('tag', 'Unknown')
            domain = res.get('domain', 'unknown')
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        logger.info(f"Tag distribution: {dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5])}")
        logger.info(f"Top domains: {dict(sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:5])}")
        
        return len(resource_dicts)
        
    except Exception as e:
        logger.error(f"Failed to sync resources: {e}")
        raise
    finally:
        session.close()

def check_resource_sync_needed(json_path: str = "data/resources/detected_resources.json") -> bool:
    """
    Check if resource sync is needed by comparing database timestamp with JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        True if sync is needed, False otherwise
    """
    if not os.path.exists(json_path):
        logger.info("JSON file doesn't exist, sync needed")
        return True
    
    # Get JSON file modification time
    json_mtime = datetime.fromtimestamp(os.path.getmtime(json_path))
    
    # Get latest resource timestamp from database
    session = SessionLocal()
    try:
        latest_resource = session.query(Resource).order_by(Resource.timestamp.desc()).first()
        if not latest_resource:
            logger.info("No resources in database")
            return False
        
        latest_db_time = latest_resource.timestamp
        
        # Compare times (with 1 minute buffer to account for processing time)
        time_diff = (latest_db_time - json_mtime).total_seconds()
        sync_needed = time_diff > 60  # More than 1 minute difference
        
        if sync_needed:
            logger.info(f"Sync needed: latest resource {latest_db_time} > JSON file {json_mtime}")
        else:
            logger.info(f"Sync not needed: JSON file is current (latest resource: {latest_db_time})")
        
        return sync_needed
        
    except Exception as e:
        logger.error(f"Failed to check sync status: {e}")
        return True  # Err on the side of syncing
    finally:
        session.close()

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sync Discord resources from database to JSON')
    parser.add_argument('--output', type=str, default="data/resources/detected_resources.json",
                       help='Output JSON file path')
    parser.add_argument('--force', action='store_true', 
                       help='Force sync even if not needed')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check if sync is needed, don\'t perform sync')
    
    args = parser.parse_args()
    
    try:
        if args.check_only:
            needed = check_resource_sync_needed(args.output)
            print(f"Sync needed: {needed}")
            return
        
        if not args.force and not check_resource_sync_needed(args.output):
            print("✅ Resource JSON is up to date, no sync needed")
            return
        
        count = sync_resources_to_json(args.output)
        print(f"✅ Successfully synced {count} resources to {args.output}")
        
    except Exception as e:
        print(f"❌ Resource sync failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()