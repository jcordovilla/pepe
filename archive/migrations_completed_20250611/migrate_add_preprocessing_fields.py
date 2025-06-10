#!/usr/bin/env python3
"""
Database Migration Script: Add Preprocessing Fields

This script adds the missing preprocessing fields to the Message table
to support enhanced weekly digest and content analysis capabilities.

New fields:
- enhanced_content: AI-enhanced content with context
- topics: extracted topics/themes
- keywords: extracted keywords/entities  
- intent: message intent classification
- sentiment: sentiment analysis result
- engagement_score: engagement metrics
- content_type: content type classification
- mentioned_technologies: technical terms mentioned
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from db.db import engine, get_db_session
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_preprocessing_columns():
    """Add preprocessing columns to the messages table."""
    
    columns_to_add = [
        ("enhanced_content", "TEXT"),
        ("topics", "JSON"),
        ("keywords", "JSON"),
        ("intent", "VARCHAR(50)"),
        ("sentiment", "VARCHAR(20)"),
        ("engagement_score", "JSON"),
        ("content_type", "VARCHAR(50)"),
        ("mentioned_technologies", "JSON")
    ]
    
    with get_db_session() as session:
        for column_name, column_type in columns_to_add:
            try:
                # Check if column already exists
                result = session.execute(text(f"PRAGMA table_info(messages)"))
                existing_columns = [row[1] for row in result.fetchall()]
                
                if column_name not in existing_columns:
                    logger.info(f"Adding column: {column_name}")
                    session.execute(text(f"ALTER TABLE messages ADD COLUMN {column_name} {column_type}"))
                    session.commit()
                else:
                    logger.info(f"Column {column_name} already exists, skipping")
                    
            except Exception as e:
                logger.error(f"Failed to add column {column_name}: {e}")
                session.rollback()
                raise

def verify_columns():
    """Verify that all preprocessing columns were added successfully."""
    expected_columns = [
        "enhanced_content", "topics", "keywords", "intent", 
        "sentiment", "engagement_score", "content_type", "mentioned_technologies"
    ]
    
    with get_db_session() as session:
        result = session.execute(text("PRAGMA table_info(messages)"))
        existing_columns = [row[1] for row in result.fetchall()]
        
        missing_columns = [col for col in expected_columns if col not in existing_columns]
        
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            return False
        else:
            logger.info("✅ All preprocessing columns successfully added!")
            return True

def get_table_info():
    """Get current table structure for verification."""
    with get_db_session() as session:
        result = session.execute(text("PRAGMA table_info(messages)"))
        columns = result.fetchall()
        
        print("\n📊 Current Message Table Structure:")
        print("-" * 60)
        for col in columns:
            col_id, name, data_type, not_null, default, pk = col
            print(f"{name:25} | {data_type:15} | {'NOT NULL' if not_null else 'NULL':8}")

def main():
    """Main migration execution."""
    print("🚀 Starting Database Migration: Add Preprocessing Fields")
    print("=" * 60)
    
    try:
        # Show current table structure
        print("\n📋 Before Migration:")
        get_table_info()
        
        # Add preprocessing columns
        print("\n🔧 Adding preprocessing columns...")
        add_preprocessing_columns()
        
        # Verify migration
        print("\n✅ Verifying migration...")
        success = verify_columns()
        
        # Show updated table structure
        print("\n📋 After Migration:")
        get_table_info()
        
        if success:
            print("\n🎉 Migration completed successfully!")
            print("The database now supports enhanced preprocessing capabilities.")
            return 0
        else:
            print("\n❌ Migration failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    exit(main())
