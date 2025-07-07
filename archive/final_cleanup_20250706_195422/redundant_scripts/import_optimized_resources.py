#!/usr/bin/env python3
"""
Import Optimized Resources - Replace Legacy Data

Replaces the low-quality legacy resource data with our 
curated, high-quality optimized resources.
"""

import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def import_optimized_resources():
    """Replace legacy junk data with optimized high-quality resources"""
    
    print("🔄 Importing Optimized High-Quality Resources")
    print("=" * 60)
    
    # Load optimized resources
    optimized_file = project_root / 'data' / 'optimized_fresh_resources.json'
    if not optimized_file.exists():
        print(f"❌ Optimized resources file not found: {optimized_file}")
        return False
    
    with open(optimized_file, 'r') as f:
        data = json.load(f)
    
    resources = data['resources']
    stats = data['statistics']
    
    print(f"📊 Optimized Resources to Import:")
    print(f"   Total resources: {len(resources):,}")
    print(f"   Quality distribution: {stats['quality_distribution']}")
    print(f"   Categories: {len(stats['categories'])} types")
    
    # Database operations
    db_path = project_root / 'data' / 'enhanced_resources.db'
    
    if not db_path.exists():
        print(f"❌ Enhanced resources database not found: {db_path}")
        return False
    
    print(f"\n🗄️ Database Operations:")
    
    # Create backup
    backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    import shutil
    shutil.copy2(db_path, backup_path)
    print(f"   💾 Backup created: {backup_path}")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get current count
    cursor.execute("SELECT COUNT(*) FROM resources")
    old_count = cursor.fetchone()[0]
    print(f"   📊 Current resources in DB: {old_count:,}")
    
    # Clear existing resources (replace with optimized)
    print(f"   🧹 Clearing legacy resources...")
    cursor.execute("DELETE FROM resources")
    cursor.execute("DELETE FROM resource_validation_history")
    cursor.execute("DELETE FROM resource_metadata")
    cursor.execute("DELETE FROM resource_relationships")
    cursor.execute("DELETE FROM resource_usage_analytics")
    cursor.execute("DELETE FROM resource_quality_history")
    
    # Reset auto-increment
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='resources'")
    
    print(f"   ✅ Cleared {old_count:,} legacy resources")
    
    # Import optimized resources
    print(f"\n📥 Importing Optimized Resources:")
    
    imported_count = 0
    skipped_count = 0
    
    for i, resource in enumerate(resources):
        try:
            # Insert resource
            cursor.execute("""
                INSERT INTO resources (
                    id, url, domain, title, description, quality_score,
                    quality_level, message_id, channel_id, resource_type,
                    author, created_at, validation_status, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"opt_{i+1}",  # Generate unique ID
                resource['url'],
                resource['domain'],
                resource.get('context', 'Imported Resource')[:100],  # Use context as title
                resource.get('content_preview', '')[:500],
                resource['quality_score'],
                _score_to_quality_level(resource['quality_score']),
                resource.get('message_id'),
                resource.get('channel_id', 'imported'),  # Default for imported resources
                resource['category'],  # Map to resource_type
                resource.get('author'),
                resource.get('timestamp'),
                'HEALTHY',  # Default validation status
                True  # Active resource
            ))
            
            imported_count += 1
            
            if (i + 1) % 50 == 0:
                print(f"   Progress: {i + 1}/{len(resources)} ({((i + 1)/len(resources)*100):.1f}%)")
        
        except Exception as e:
            print(f"   ⚠️ Error importing resource {i+1}: {e}")
            skipped_count += 1
            continue
    
    # Commit changes
    conn.commit()
    
    # Verify import
    cursor.execute("SELECT COUNT(*) FROM resources")
    new_count = cursor.fetchone()[0]
    
    # Get quality stats
    cursor.execute("""
        SELECT 
            AVG(quality_score) as avg_quality,
            COUNT(CASE WHEN quality_score >= 0.9 THEN 1 END) as excellent,
            COUNT(CASE WHEN quality_score >= 0.8 AND quality_score < 0.9 THEN 1 END) as high,
            COUNT(CASE WHEN quality_score >= 0.7 AND quality_score < 0.8 THEN 1 END) as good
        FROM resources
    """)
    quality_stats = cursor.fetchone()
    
    # Get top domains
    cursor.execute("""
        SELECT domain, COUNT(*) as count 
        FROM resources 
        GROUP BY domain 
        ORDER BY count DESC 
        LIMIT 10
    """)
    top_domains = cursor.fetchall()
    
    conn.close()
    
    print(f"\n🎉 Import Complete!")
    print(f"   ✅ Successfully imported: {imported_count:,} resources")
    print(f"   ⚠️ Skipped: {skipped_count} resources")
    print(f"   📊 Total in database: {new_count:,} resources")
    
    if quality_stats[0] is not None:
        print(f"   📈 Average quality score: {quality_stats[0]:.3f}")
        print(f"   ⭐ Quality distribution:")
        print(f"      Excellent (≥0.9): {quality_stats[1]} resources")
        print(f"      High (0.8-0.9): {quality_stats[2]} resources") 
        print(f"      Good (0.7-0.8): {quality_stats[3]} resources")
    else:
        print(f"   📈 No quality stats available (empty database)")
    
    print(f"\n🌐 Top Domains Now:")
    for domain, count in top_domains[:5]:
        print(f"   {domain}: {count} resources")
    
    print(f"\n💡 Result:")
    if quality_stats[0] is not None and new_count > 0:
        quality_percentage = ((quality_stats[1] + quality_stats[2]) / new_count * 100)
        
        if quality_percentage >= 60:
            print(f"🎉 SUCCESS! {quality_percentage:.1f}% high-quality resources!")
            print("✅ Your resource database now contains curated, valuable content.")
            print("✅ No more LinkedIn profiles, Discord CDN links, or expired Zoom meetings!")
        else:
            print(f"⚠️ Quality percentage: {quality_percentage:.1f}%")
    else:
        print(f"⚠️ No resources imported successfully")
    
    return True

def _score_to_quality_level(score: float) -> str:
    """Convert quality score to quality level string"""
    if score >= 0.9:
        return "EXCELLENT"
    elif score >= 0.8:
        return "HIGH"
    elif score >= 0.7:
        return "GOOD"
    elif score >= 0.5:
        return "FAIR"
    else:
        return "POOR"

async def main():
    """Main import execution"""
    try:
        success = await import_optimized_resources()
        
        if success:
            print(f"\n🚀 Next Steps:")
            print(f"   1. Test resource search: poetry run ./pepe-admin stats")
            print(f"   2. Start bot for real-time resource detection: poetry run python main.py")
            print(f"   3. Future resources will be added incrementally (no full replacement)")
            return 0
        else:
            print(f"\n❌ Import failed")
            return 1
    
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 