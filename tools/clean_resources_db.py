"""
Cleans the resources table in-place by re-enriching, deduplicating, and re-classifying all resources using the latest pipeline logic.
This will run the enrich and dedup logic on all resources in the database, and then update the database with the cleaned resources.
Careful, it will take a while to run, and will modify the database in-place.
"""
import sys
import json
from db.db import SessionLocal, Resource
from core.resource_detector import ai_enrich_title_description, needs_title_fix
from core.classifier import classify_resource  # If you use a classifier
from core.resource_detector import deduplicate_resources  # Use the same dedup logic
from tqdm import tqdm


def main():
    session = SessionLocal()
    try:
        resources = session.query(Resource).all()
        print(f"Loaded {len(resources)} resources from DB.")
        # Convert to dicts for easier processing
        resource_dicts = []
        for res in resources:
            resource_dicts.append({
                'db_obj': res,  # Keep reference for updating later
                'id': res.id,
                'name': res.name,
                'description': res.description,
                'url': res.url,
                'type': res.type,
                'tag': res.tag,
                'author': res.author,
                'author_display': res.author_display,
                'channel_name': res.channel_name,
                'timestamp': res.timestamp,
                'context_snippet': res.context_snippet,
                'jump_url': getattr(res, 'jump_url', None),
                'meta': res.meta,
                'message_id': res.message_id,
                'guild_id': res.guild_id,
                'channel_id': res.channel_id,
            })

        # Re-enrich titles/descriptions
        for r in tqdm(resource_dicts, desc="Enriching resources"):
            if needs_title_fix(r):
                title, description = ai_enrich_title_description(r)
                r['name'] = title
                r['description'] = description

        # Re-classify/tag if needed
        for r in tqdm(resource_dicts, desc="Classifying resources"):
            if not r['tag'] or r['tag'] == 'Unknown':
                r['tag'] = classify_resource(r['url'], r['name'], r['description'])

        # Deduplicate using the same logic as the pipeline
        deduped = deduplicate_resources(resource_dicts)
        print(f"Deduplicated: {len(resource_dicts)} -> {len(deduped)}")

        # Update DB: Overwrite improved fields, remove duplicates
        dedup_ids = set(r['id'] for r in deduped)
        for r in resource_dicts:
            db_obj = r['db_obj']
            if r['id'] not in dedup_ids:
                session.delete(db_obj)
                continue
            # Overwrite with improved values
            db_obj.name = r['name']
            db_obj.description = r['description']
            db_obj.tag = r['tag']
            # Optionally update other fields if needed
            session.add(db_obj)
        session.commit()
        print("Database cleaned and updated.")
    finally:
        session.close()

if __name__ == "__main__":
    main()
