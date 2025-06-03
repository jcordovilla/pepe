#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.tools import get_channels

try:
    channels = get_channels()
    print('Available channels:')
    for channel in channels:
        print(f'- {channel["name"]} (ID: {channel["id"]}, Messages: {channel["message_count"]})')
    print(f'\nTotal channels: {len(channels)}')
    
    # Look for agent-dev channel specifically
    agent_dev_channels = [ch for ch in channels if 'agent' in ch['name'].lower() and 'dev' in ch['name'].lower()]
    if agent_dev_channels:
        print(f'\nAgent dev channels found:')
        for ch in agent_dev_channels:
            print(f'- {ch["name"]} (ID: {ch["id"]}, Messages: {ch["message_count"]})')
    else:
        print('\nNo agent-dev like channels found')
        
    # Look for any channels with "agent" in the name
    agent_channels = [ch for ch in channels if 'agent' in ch['name'].lower()]
    if agent_channels:
        print(f'\nChannels with "agent" in name:')
        for ch in agent_channels:
            print(f'- {ch["name"]} (ID: {ch["id"]}, Messages: {ch["message_count"]})')
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
