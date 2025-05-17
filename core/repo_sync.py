# Project: resources-library
# Description: Syncs resources from the database to Markdown files
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import yaml  # Requires PyYAML
from db.db import Resource
import json

def sync_to_markdown(db_url: str, output_dir: str = "docs/resources"):
    """
    Connect to the database at db_url, fetch all Resource records,
    and write one Markdown file per resource under output_dir,
    with YAML front-matter (title, date, author, channel, tag, original_url, description)
    followed by the description as the main content.
    """
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    os.makedirs(output_dir, exist_ok=True)
    resources = session.query(Resource).all()

    for res in resources:
        # Use the correct name and description fields from the database
        title = res.name or res.url
        description = res.description or (res.context_snippet or "")
        front_matter = {
            "title": title,
            "date": res.timestamp.strftime("%Y-%m-%d") if res.timestamp else None,
            "author": res.author_display or res.author,
            "channel": res.channel_name or res.channel_id,
            "tag": res.tag,
            "original_url": res.url,
            "description": description,
        }
        filename = f"{res.id}.md"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as md_file:
            md_file.write("---\n")
            yaml.dump(front_matter, md_file, default_flow_style=False, sort_keys=False)
            md_file.write("---\n\n")
            md_file.write(description)

    print(f"Wrote {len(resources)} Markdown files to {output_dir}")


def sync_to_json(db_url: str, output_path: str = "docs/resources/resources.json"):
    """
    Connect to the database at db_url, fetch all Resource records,
    and write them all to a single JSON file at output_path.
    """
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    resources = session.query(Resource).all()

    resource_dicts = []
    for res in resources:
        resource_dicts.append({
            "id": res.id,
            "title": res.name or res.url,  # Use the correct name field
            "description": res.description or (res.context_snippet or ""),  # Use the correct description field
            "date": res.timestamp.strftime("%Y-%m-%d") if res.timestamp else None,
            "author": res.author_display or res.author,
            "channel": res.channel_name or res.channel_id,
            "tag": res.tag,
            "resource_url": res.url,  # Renamed from original_url
            "discord_url": getattr(res, 'jump_url', None),  # Renamed from jump_url
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(resource_dicts, f, indent=2, default=str)

    print(f"Wrote {len(resource_dicts)} resources to {output_path}")

if __name__ == "__main__":
    # Only output a single JSON file, not individual JSONs per resource
    sync_to_json("sqlite:///data/discord_messages.db")
