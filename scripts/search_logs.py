#!/usr/bin/env python3
"""
Simple script to search through query logs for easy content discovery.

Usage:
    python search_logs.py "search term"
    python search_logs.py --interface discord "AI"
    python search_logs.py --user "john" 
    python search_logs.py --date 20250609
"""

import argparse
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
import json

def search_simple_logs(search_term=None, interface=None, user_filter=None, date_filter=None, days_back=7):
    """Search through simple text logs."""
    
    log_dir = Path("logs/simple_logs")
    if not log_dir.exists():
        print("❌ No simple logs directory found. Make sure the logging system is working.")
        return
    
    # Get log files to search
    log_files = []
    if date_filter:
        # Search specific date
        log_file = log_dir / f"queries_simple_{date_filter}.txt"
        if log_file.exists():
            log_files.append(log_file)
    else:
        # Search recent files
        today = datetime.now()
        for i in range(days_back):
            date = (today - timedelta(days=i)).strftime('%Y%m%d')
            log_file = log_dir / f"queries_simple_{date}.txt"
            if log_file.exists():
                log_files.append(log_file)
    
    if not log_files:
        print(f"❌ No log files found for the specified criteria.")
        return
    
    print(f"🔍 Searching {len(log_files)} log files...")
    
    matches = []
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Split into individual entries
                entries = content.split("=== QUERY LOG ENTRY ===")
                
                for entry in entries[1:]:  # Skip first empty split
                    if not entry.strip():
                        continue
                    
                    # Apply filters
                    if interface and f"Interface: {interface.upper()}" not in entry:
                        continue
                    
                    if user_filter and user_filter.lower() not in entry.lower():
                        continue
                    
                    if search_term and search_term.lower() not in entry.lower():
                        continue
                    
                    matches.append({
                        'file': log_file.name,
                        'entry': entry.strip()
                    })
        
        except Exception as e:
            print(f"⚠️ Error reading {log_file}: {e}")
    
    # Display results
    if not matches:
        print("❌ No matches found.")
        return
    
    print(f"\n✅ Found {len(matches)} matching entries:\n")
    
    for i, match in enumerate(matches, 1):
        print(f"{'='*60}")
        print(f"MATCH #{i} (from {match['file']})")
        print(f"{'='*60}")
        print(match['entry'])
        print()

def search_json_logs(search_term=None, interface=None, user_filter=None, date_filter=None, days_back=7):
    """Search through JSON logs."""
    
    log_dir = Path("logs/query_logs")
    if not log_dir.exists():
        print("❌ No JSON logs directory found.")
        return
    
    # Get log files to search
    log_files = []
    if date_filter:
        log_file = log_dir / f"queries_{date_filter}.jsonl"
        if log_file.exists():
            log_files.append(log_file)
    else:
        today = datetime.now()
        for i in range(days_back):
            date = (today - timedelta(days=i)).strftime('%Y%m%d')
            log_file = log_dir / f"queries_{date}.jsonl"
            if log_file.exists():
                log_files.append(log_file)
    
    if not log_files:
        print(f"❌ No JSON log files found.")
        return
    
    print(f"🔍 Searching {len(log_files)} JSON log files...")
    
    matches = []
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        entry = json.loads(line)
                        
                        # Apply filters
                        if interface and entry.get('routing_strategy') != interface:
                            continue
                        
                        if user_filter and user_filter.lower() not in entry.get('username', '').lower():
                            continue
                        
                        if search_term:
                            text_to_search = f"{entry.get('query_text', '')} {entry.get('response_text', '')}"
                            if search_term.lower() not in text_to_search.lower():
                                continue
                        
                        matches.append({
                            'file': log_file.name,
                            'line': line_num,
                            'entry': entry
                        })
                    
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            print(f"⚠️ Error reading {log_file}: {e}")
    
    # Display results
    if not matches:
        print("❌ No matches found in JSON logs.")
        return
    
    print(f"\n✅ Found {len(matches)} matching JSON entries:\n")
    
    for i, match in enumerate(matches, 1):
        entry = match['entry']
        print(f"{'='*60}")
        print(f"JSON MATCH #{i} (from {match['file']}:{match['line']})")
        print(f"{'='*60}")
        print(f"Timestamp: {entry.get('timestamp', 'N/A')}")
        print(f"User: {entry.get('username', 'N/A')} (ID: {entry.get('user_id', 'N/A')})")
        print(f"Strategy: {entry.get('routing_strategy', 'N/A')}")
        print(f"Query: {entry.get('query_text', 'N/A')}")
        print(f"Status: {entry.get('response_status', 'N/A')}")
        print(f"Processing Time: {entry.get('processing_time_ms', 'N/A')}ms")
        if entry.get('response_text') and len(entry['response_text']) < 200:
            print(f"Response: {entry['response_text']}")
        elif entry.get('response_text'):
            print(f"Response: {entry['response_text'][:200]}...")
        print()

def main():
    parser = argparse.ArgumentParser(description='Search through query logs')
    parser.add_argument('search_term', nargs='?', help='Text to search for in queries and responses')
    parser.add_argument('--interface', choices=['discord', 'streamlit'], help='Filter by interface')
    parser.add_argument('--user', help='Filter by username (partial match)')
    parser.add_argument('--date', help='Search specific date (YYYYMMDD format)')
    parser.add_argument('--days', type=int, default=7, help='Number of days back to search (default: 7)')
    parser.add_argument('--format', choices=['simple', 'json', 'both'], default='simple', 
                       help='Log format to search (default: simple)')
    parser.add_argument('--list-files', action='store_true', help='List available log files')
    
    args = parser.parse_args()
    
    if args.list_files:
        print("📁 Available log files:")
        print("\n📄 Simple logs:")
        simple_dir = Path("logs/simple_logs")
        if simple_dir.exists():
            for log_file in sorted(simple_dir.glob("*.txt")):
                print(f"  - {log_file.name}")
        else:
            print("  No simple logs directory found")
        
        print("\n📊 JSON logs:")
        json_dir = Path("logs/query_logs")
        if json_dir.exists():
            for log_file in sorted(json_dir.glob("*.jsonl")):
                print(f"  - {log_file.name}")
        else:
            print("  No JSON logs directory found")
        return
    
    if not args.search_term and not args.interface and not args.user:
        print("❌ Please provide a search term, interface filter, or user filter.")
        parser.print_help()
        return
    
    print(f"🔍 Searching logs with filters:")
    if args.search_term:
        print(f"  📝 Search term: '{args.search_term}'")
    if args.interface:
        print(f"  🖥️  Interface: {args.interface}")
    if args.user:
        print(f"  👤 User filter: '{args.user}'")
    if args.date:
        print(f"  📅 Date: {args.date}")
    else:
        print(f"  📅 Days back: {args.days}")
    print()
    
    if args.format in ['simple', 'both']:
        search_simple_logs(
            search_term=args.search_term,
            interface=args.interface,
            user_filter=args.user,
            date_filter=args.date,
            days_back=args.days
        )
    
    if args.format in ['json', 'both']:
        search_json_logs(
            search_term=args.search_term,
            interface=args.interface,
            user_filter=args.user,
            date_filter=args.date,
            days_back=args.days
        )

if __name__ == "__main__":
    main()
