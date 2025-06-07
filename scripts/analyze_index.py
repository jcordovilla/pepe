#!/usr/bin/env python3
"""
Analyze the current FAISS index and metadata structure
"""

import pickle
import faiss
import numpy as np
from collections import Counter
import json

def analyze_faiss_index():
    """Analyze the current FAISS index and metadata"""
    
    # Load FAISS index
    try:
        index = faiss.read_index("index_faiss/index.faiss")
        print(f"=== FAISS Index Analysis ===")
        print(f"Index type: {type(index)}")
        print(f"Total vectors: {index.ntotal}")
        print(f"Vector dimension: {index.d}")
        print(f"Is trained: {index.is_trained}")
        print()
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return
    
    # Load metadata
    try:
        with open("index_faiss/index.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        print(f"=== Metadata Analysis ===")
        print(f"Metadata type: {type(metadata)}")
        
        # Handle different metadata structures
        if hasattr(metadata, '__len__'):
            print(f"Metadata length: {len(metadata)}")
            
            # Try to access as list/dict
            if hasattr(metadata, '__getitem__'):
                try:
                    sample = metadata[0] if len(metadata) > 0 else None
                    if sample:
                        print(f"Sample entry type: {type(sample)}")
                        if hasattr(sample, 'keys'):
                            print(f"Sample keys: {list(sample.keys())}")
                        elif isinstance(sample, dict):
                            print(f"Sample keys: {list(sample.keys())}")
                        print(f"Sample content preview: {str(sample)[:200]}...")
                except Exception as e:
                    print(f"Error accessing sample: {e}")
        
        print(f"Metadata structure: {str(metadata)[:500]}...")
        
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

def analyze_database_content():
    """Analyze the database content to understand what's being embedded"""
    try:
        from db.db import SessionLocal, Message
        
        session = SessionLocal()
        
        # Get total message count
        total_count = session.query(Message).count()
        print(f"\n=== Database Content Analysis ===")
        print(f"Total messages in DB: {total_count}")
        
        # Sample messages
        sample_messages = session.query(Message).limit(10).all()
        
        print(f"\n=== Sample Messages ===")
        for i, msg in enumerate(sample_messages[:3]):
            print(f"Message {i+1}:")
            print(f"  Content: {msg.content[:100]}...")
            print(f"  Channel: {msg.channel_name}")
            print(f"  Author: {msg.author}")
            print(f"  Length: {len(msg.content)} chars")
            print()
        
        # Analyze content patterns
        all_messages = session.query(Message.content, Message.channel_name).all()
        
        # Content length analysis
        lengths = [len(msg.content) for msg, _ in all_messages]
        print(f"=== Content Length Analysis ===")
        print(f"Average length: {sum(lengths) / len(lengths):.1f} chars")
        print(f"Min length: {min(lengths)} chars")
        print(f"Max length: {max(lengths)} chars")
        print(f"Median length: {sorted(lengths)[len(lengths)//2]} chars")
        
        # Channel analysis
        channels = [channel for _, channel in all_messages]
        channel_counts = Counter(channels)
        print(f"\n=== Channel Distribution ===")
        for channel, count in channel_counts.most_common(10):
            print(f"{channel}: {count} messages")
        
        # Content type analysis
        code_count = sum(1 for msg, _ in all_messages if '```' in msg or 'import ' in msg or 'def ' in msg)
        url_count = sum(1 for msg, _ in all_messages if 'http' in msg)
        short_count = sum(1 for msg, _ in all_messages if len(msg) < 50)
        long_count = sum(1 for msg, _ in all_messages if len(msg) > 500)
        
        print(f"\n=== Content Type Patterns ===")
        print(f"Messages with code: {code_count} ({code_count/len(all_messages)*100:.1f}%)")
        print(f"Messages with URLs: {url_count} ({url_count/len(all_messages)*100:.1f}%)")
        print(f"Short messages (<50 chars): {short_count} ({short_count/len(all_messages)*100:.1f}%)")
        print(f"Long messages (>500 chars): {long_count} ({long_count/len(all_messages)*100:.1f}%)")
        
        session.close()
        
    except Exception as e:
        print(f"Error analyzing database: {e}")

if __name__ == "__main__":
    analyze_faiss_index()
    analyze_database_content()
