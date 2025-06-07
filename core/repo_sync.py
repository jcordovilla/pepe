# Project: resources-library
# Description: Enhanced syncing of resources from database to Markdown files and JSON
import os
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import yaml  # Requires PyYAML

from db.db import get_db_session, Resource, execute_query
from core.resource_detector import simple_enrich_title
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enrich_resource_title_description(resource_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance resource title and description if they're missing or generic.
    Uses the same logic as our optimized resource detector.
    """
    # Check if enrichment is needed
    title = resource_data.get('title', '') or ''
    description = resource_data.get('description', '') or ''
    url = resource_data.get('resource_url', '') or resource_data.get('url', '')
    
    needs_enrichment = (
        not title or 
        title == url or
        title.endswith(' Resource') or
        len(description) < 20
    )
    
    if needs_enrichment and url:
        # Create a temporary resource dict for enrichment
        temp_resource = {
            'url': url,
            'context_snippet': resource_data.get('description', '') or ''
        }
        
        try:
            new_title, new_desc = simple_enrich_title(temp_resource)
            if not title or title == url or title.endswith(' Resource'):
                resource_data['title'] = new_title
            if len(description) < 20:
                resource_data['description'] = new_desc
        except Exception as e:
            logger.warning(f"Failed to enrich resource {url}: {e}")
    
    return resource_data

def sync_to_markdown(output_dir: str = "docs/resources", enrich_titles: bool = True) -> int:
    """
    Fetch all Resource records and write one Markdown file per resource,
    with YAML front-matter followed by the description as content.
    
    Args:
        output_dir: Directory to write markdown files
        enrich_titles: Whether to enhance missing titles/descriptions
        
    Returns:
        Number of files written
    """
    def get_all_resources(session):
        return session.query(Resource).order_by(Resource.timestamp.desc()).all()
    
    try:
        resources = execute_query(get_all_resources)
        logger.info(f"Retrieved {len(resources)} resources from database")
        
        os.makedirs(output_dir, exist_ok=True)
        written_count = 0
        
        for res in resources:
            try:
                # Prepare resource data
                resource_data = {
                    "title": res.name or res.url,
                    "description": res.description or res.context_snippet or "",
                    "resource_url": res.url
                }
                
                # Enrich if needed
                if enrich_titles:
                    resource_data = enrich_resource_title_description(resource_data)
                
                # Prepare front matter
                front_matter = {
                    "title": resource_data["title"],
                    "date": res.timestamp.strftime("%Y-%m-%d") if res.timestamp else None,
                    "author": res.author_display or res.author,
                    "channel": res.channel_name or res.channel_id,
                    "tag": res.tag,
                    "original_url": res.url,
                    "description": resource_data["description"][:200] + "..." if len(resource_data["description"]) > 200 else resource_data["description"],
                }
                
                # Write markdown file
                filename = f"{res.id}.md"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, "w", encoding="utf-8") as md_file:
                    md_file.write("---\n")
                    yaml.dump(front_matter, md_file, default_flow_style=False, sort_keys=False)
                    md_file.write("---\n\n")
                    md_file.write(resource_data["description"])
                
                written_count += 1
                
            except Exception as e:
                logger.error(f"Failed to write markdown for resource {res.id}: {e}")
                continue
        
        logger.info(f"Wrote {written_count} Markdown files to {output_dir}")
        return written_count
        
    except Exception as e:
        logger.error(f"Failed to sync to markdown: {e}")
        raise

def sync_to_json(output_path: str = "docs/resources/resources.json", 
                enrich_titles: bool = True,
                filter_tags: Optional[List[str]] = None,
                max_resources: Optional[int] = None) -> int:
    """
    Fetch all Resource records and write them to a single JSON file.
    
    Args:
        output_path: Path to write JSON file
        enrich_titles: Whether to enhance missing titles/descriptions
        filter_tags: Optional list of tags to filter by
        max_resources: Optional limit on number of resources
        
    Returns:
        Number of resources written
    """
    def get_filtered_resources(session):
        query = session.query(Resource).order_by(Resource.timestamp.desc())
        
        if filter_tags:
            query = query.filter(Resource.tag.in_(filter_tags))
        
        if max_resources:
            query = query.limit(max_resources)
            
        return query.all()
    
    try:
        resources = execute_query(get_filtered_resources)
        logger.info(f"Retrieved {len(resources)} resources from database")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        resource_dicts = []
        processed_count = 0
        
        for res in resources:
            try:
                # Prepare basic resource data
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
                    "channel_id": res.channel_id
                }
                
                # Enrich if needed
                if enrich_titles:
                    resource_data = enrich_resource_title_description(resource_data)
                
                # Add domain for easier filtering/grouping
                try:
                    parsed_url = urlparse(res.url)
                    resource_data["domain"] = parsed_url.netloc.replace('www.', '')
                except:
                    resource_data["domain"] = "unknown"
                
                resource_dicts.append(resource_data)
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process resource {res.id}: {e}")
                continue
        
        # Write JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(resource_dicts, f, indent=2, default=str)
        
        logger.info(f"Wrote {len(resource_dicts)} resources to {output_path}")
        
        # Generate summary statistics
        tag_counts = {}
        domain_counts = {}
        for res in resource_dicts:
            tag = res.get('tag', 'Unknown')
            domain = res.get('domain', 'unknown')
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        logger.info(f"Tag distribution: {dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10])}")
        logger.info(f"Top domains: {dict(sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:10])}")
        
        return len(resource_dicts)
        
    except Exception as e:
        logger.error(f"Failed to sync to JSON: {e}")
        raise

def sync_filtered_resources(tag_filter: Optional[str] = None, 
                          output_path: Optional[str] = None) -> int:
    """
    Sync only resources with specific tags for curated lists.
    
    Args:
        tag_filter: Tag to filter by (e.g., 'Paper', 'Tool')
        output_path: Custom output path
        
    Returns:
        Number of resources synced
    """
    if not output_path:
        tag_safe = tag_filter.lower().replace('/', '_') if tag_filter else 'all'
        output_path = f"docs/resources/resources_{tag_safe}.json"
    
    filter_tags = [tag_filter] if tag_filter else None
    
    return sync_to_json(
        output_path=output_path,
        enrich_titles=True,
        filter_tags=filter_tags
    )

if __name__ == "__main__":
    """
    Main execution with multiple sync options.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Sync Discord resources to JSON/Markdown')
    parser.add_argument('--format', choices=['json', 'markdown', 'both'], default='json',
                       help='Output format (default: json)')
    parser.add_argument('--output', type=str, help='Output path (optional)')
    parser.add_argument('--tag', type=str, help='Filter by tag (optional)')
    parser.add_argument('--max', type=int, help='Maximum number of resources (optional)')
    parser.add_argument('--no-enrich', action='store_true', help='Skip title/description enrichment')
    
    args = parser.parse_args()
    
    try:
        if args.format in ['json', 'both']:
            output_path = args.output or "docs/resources/resources.json"
            if args.tag:
                count = sync_filtered_resources(args.tag, output_path)
            else:
                count = sync_to_json(
                    output_path=output_path,
                    enrich_titles=not args.no_enrich,
                    max_resources=args.max
                )
            print(f"✅ JSON sync completed: {count} resources")
        
        if args.format in ['markdown', 'both']:
            output_dir = args.output or "docs/resources"
            count = sync_to_markdown(
                output_dir=output_dir,
                enrich_titles=not args.no_enrich
            )
            print(f"✅ Markdown sync completed: {count} files")
            
    except Exception as e:
        print(f"❌ Sync failed: {e}")
        exit(1)
