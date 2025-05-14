# Project: resources-library
# Description: Syncs resources from the database to Markdown files
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import yaml  # Requires PyYAML
from db.db import Resource

def sync_to_markdown(db_url: str, output_dir: str = "docs/resources"):
    """
    Connect to the database at db_url, fetch all Resource records,
    and write one Markdown file per resource under output_dir,
    with YAML front-matter (title, date, author, channel, tag, original_url)
    followed by the context_snippet.
    """
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    os.makedirs(output_dir, exist_ok=True)
    resources = session.query(Resource).all()

    for res in resources:
        # Use new fields if available, fallback to old
        title = getattr(res, 'name', None) or (getattr(res, 'meta', {}) or {}).get('title') or res.url
        description = getattr(res, 'description', None) or (getattr(res, 'meta', {}) or {}).get('description') or (res.context_snippet or "")
        front_matter = {
            "title": title,
            "date": res.timestamp.strftime("%Y-%m-%d") if res.timestamp else None,
            "author": res.author_display or res.author,
            "channel": res.channel_name or res.channel_id,
            "tag": res.tag,
            "original_url": res.url,
        }
        filename = f"{res.id}.md"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as md_file:
            md_file.write("---\n")
            yaml.dump(front_matter, md_file, default_flow_style=False, sort_keys=False)
            md_file.write("---\n\n")
            md_file.write(description)

    print(f"Wrote {len(resources)} Markdown files to {output_dir}")

if __name__ == "__main__":
    sync_to_markdown("sqlite:///resources.db")
