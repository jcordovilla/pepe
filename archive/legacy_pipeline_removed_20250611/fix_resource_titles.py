# Script to detect resources with a URL as title and message content as description, and use AI to generate better title/description
import json
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.ai_client import AIClient
from tqdm import tqdm

# Initialize local AI client
ai_client = AIClient()

INPUT_PATH = "data/resources/detected_resources.json"
OUTPUT_PATH = "docs/resources/resources_fixed.json"

def is_url(text):
    return isinstance(text, str) and text.startswith("http")

def needs_fix(resource):
    return is_url(resource.get("title", "")) and resource.get("description", "").strip() != ""

def ai_enrich_title_description(resource):
    prompt = f"""
Given the following Discord message content and resource URL, generate a concise, human-readable title and a 1-2 sentence description suitable for a public resource library. Do not use the raw URL as the title. If the message is just a link, infer the likely topic from the URL or context.

Message content:
{resource['description']}

Resource URL:
{resource['resource_url']}

Respond in JSON with keys 'title' and 'description'.
"""
    response = ai_client.chat_completion([{"role": "user", "content": prompt}])
    try:
        data = json.loads(response)
        return data["title"], data["description"]
    except Exception:
        # fallback: just use the first 10 words of the message as title
        desc = resource["description"]
        words = desc.split()
        return ("Resource: " + " ".join(words[:10]), desc)

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        resources = json.load(f)

    updated = 0
    to_fix = [r for r in resources if needs_fix(r)]
    print(f"Found {len(to_fix)} resources needing AI enrichment.")
    
    for resource in tqdm(to_fix, desc="Enriching resources", unit="msg"):
        title, description = ai_enrich_title_description(resource)
        resource["title"] = title
        resource["description"] = description
        updated += 1

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(resources, f, indent=2, ensure_ascii=False)

    print(f"Updated {updated} resources. Output written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
