#!/usr/bin/env python3
"""
dedupe_discord_resources.py: Deduplicate a JSON list of Discord message entries based on resource_url or title.
Usage:
    python dedupe_discord_resources.py <source.json>
"""
import os
import sys
import json
import argparse
import re
from urllib.parse import urlparse, urlunparse

def normalize_url(url: str) -> str:
    """Strip query parameters, fragments, and trailing punctuation from a URL."""
    try:
        parsed = urlparse(url)
        parsed = parsed._replace(query='', fragment='')
        clean = urlunparse(parsed).rstrip(') ').strip()
        return clean
    except Exception:
        return url.strip()

def normalize_title(title: str) -> str:
     """Lowercase, remove non-alphanumeric chars, and collapse whitespace in titles."""
     t = title.lower()
     t = re.sub(r'[^a-z0-9\s]', '', t)
     t = re.sub(r'\s+', ' ', t).strip()
     return t

def main():
     parser = argparse.ArgumentParser(
         description='Deduplicate Discord JSON by resource_url or title.'
     )
     parser.add_argument(
         'input_file',
         help='Path to source JSON file (a list of message entries)'
     )
     args = parser.parse_args()

     input_path = args.input_file
     if not os.path.isfile(input_path):
         print(f"Error: Input file not found: {input_path}", file=sys.stderr)
         sys.exit(1)

     try:
         with open(input_path, 'r', encoding='utf-8') as f:
             data = json.load(f)
     except json.JSONDecodeError as e:
         print(f"Error: Failed to parse JSON: {e}", file=sys.stderr)
         sys.exit(1)
     except Exception as e:
         print(f"Error reading file: {e}", file=sys.stderr)
         sys.exit(1)

     total = len(data)
     seen_urls = set()
     seen_titles = set()
     unique = []
     duplicates = []

     for idx, item in enumerate(data, start=1):
         url = item.get('resource_url')
         title = item.get('title', '')
         norm_url = normalize_url(url) if url else None
         norm_title = normalize_title(title) if title else None

         # If we've seen this URL or title before, it's a duplicate
         if (norm_url and norm_url in seen_urls) or (norm_title and norm_title in seen_titles):
             duplicates.append(item)
         else:
             unique.append(item)
             if norm_url:
                 seen_urls.add(norm_url)
             if norm_title:
                 seen_titles.add(norm_title)

     base, ext = os.path.splitext(input_path)
     cleaned_path = f"{base}.cleaned{ext}"
     duplicates_path = f"{base}.duplicates{ext}"

     try:
         with open(cleaned_path, 'w', encoding='utf-8') as f:
             json.dump(unique, f, ensure_ascii=False, indent=2)
         with open(duplicates_path, 'w', encoding='utf-8') as f:
             json.dump(duplicates, f, ensure_ascii=False, indent=2)
     except Exception as e:
         print(f"Error writing output files: {e}", file=sys.stderr)
         sys.exit(1)

     # Final summary
     print(f"Loaded {total} items from {input_path}")
     print(f"Identified {len(duplicates)} duplicates")
     print(f"Wrote {len(unique)} unique entries to {cleaned_path}")
     print(f"Wrote {len(duplicates)} duplicates to {duplicates_path}")

if __name__ == '__main__':
     main()
