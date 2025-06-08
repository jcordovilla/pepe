#!/usr/bin/env python3
"""
Enhanced Weekly Digest Implementation

This script creates a comprehensive weekly digest that:
1. Fixes timeframe issues by using proper temporal filtering
2. Creates structured, formatted digest output
3. Leverages existing message data effectively (without requiring preprocessing fields)
4. Uses statistical analysis for meaningful insights
5. Implements proper channel and user coverage
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Set
from collections import defaultdict, Counter
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import get_agent_answer
from db import SessionLocal, Message
from sqlalchemy import func, and_, desc
from core.ai_client import get_ai_client

class EnhancedWeeklyDigest:
    """Enhanced weekly digest generator that addresses identified issues."""
    
    def __init__(self):
        self.db = SessionLocal()
        self.ai_client = get_ai_client()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()
        
    def get_week_timeframe(self, days_back: int = 7) -> tuple:
        """Get precise timeframe for analysis."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        return start_date, end_date
        
    def analyze_week_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Comprehensive analysis of weekly data."""
        print(f"🔍 Analyzing week data: {start_date.date()} to {end_date.date()}")
        
        # Get all messages from the timeframe
        messages = self.db.query(Message).filter(
            and_(
                Message.timestamp >= start_date,
                Message.timestamp <= end_date
            )
        ).order_by(desc(Message.timestamp)).all()
        
        if not messages:
            return {"total_messages": 0, "error": "No messages in timeframe"}
            
        # Statistical analysis
        channel_stats = Counter()
        user_stats = Counter()
        daily_stats = Counter()
        hourly_stats = Counter()
        content_length_stats = []
        reaction_stats = Counter()
        
        # Content analysis without preprocessing fields
        word_frequency = Counter()
        emoji_usage = Counter()
        link_count = 0
        long_messages = []
        active_threads = set()
        
        for msg in messages:
            # Basic stats
            channel_name = msg.channel_name or f"unknown_{msg.channel_id}"
            channel_stats[channel_name] += 1
            
            # User stats
            author_data = msg.author if isinstance(msg.author, dict) else {}
            author_name = author_data.get("username", "unknown") if author_data else "unknown"
            user_stats[author_name] += 1
            
            # Temporal stats
            day_key = msg.timestamp.date().isoformat()
            daily_stats[day_key] += 1
            hourly_stats[msg.timestamp.hour] += 1
            
            # Content analysis
            content = msg.content or ""
            content_length_stats.append(len(content))
            
            # Word frequency (simple approach)
            words = content.lower().split()
            for word in words:
                if len(word) > 3 and word.isalpha():  # Filter meaningful words
                    word_frequency[word] += 1
                    
            # Emoji detection
            for char in content:
                if ord(char) > 127000:  # Simple emoji detection
                    emoji_usage[char] += 1
                    
            # Link detection
            if "http" in content:
                link_count += 1
                
            # Long message detection
            if len(content) > 500:
                long_messages.append({
                    "author": author_name,
                    "channel": channel_name,
                    "length": len(content),
                    "preview": content[:200] + "...",
                    "timestamp": msg.timestamp.isoformat()
                })
                
            # Reaction analysis
            try:
                reactions = msg.reactions if isinstance(msg.reactions, list) else []
                for reaction in reactions:
                    if isinstance(reaction, dict) and "emoji" in reaction:
                        emoji = reaction["emoji"]
                        count = reaction.get("count", 1)
                        reaction_stats[emoji] += count
            except:
                pass
                
        # Calculate insights
        avg_message_length = sum(content_length_stats) / len(content_length_stats) if content_length_stats else 0
        peak_hour = hourly_stats.most_common(1)[0] if hourly_stats else (0, 0)
        most_active_day = daily_stats.most_common(1)[0] if daily_stats else ("unknown", 0)
        
        return {
            "timeframe": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": (end_date - start_date).days
            },
            "overview": {
                "total_messages": len(messages),
                "active_channels": len(channel_stats),
                "active_users": len(user_stats),
                "avg_message_length": round(avg_message_length, 1),
                "total_links_shared": link_count,
                "total_reactions": sum(reaction_stats.values())
            },
            "top_channels": dict(channel_stats.most_common(10)),
            "top_users": dict(user_stats.most_common(10)),
            "daily_activity": dict(daily_stats),
            "peak_activity": {
                "hour": peak_hour[0],
                "messages": peak_hour[1],
                "day": most_active_day[0],
                "day_messages": most_active_day[1]
            },
            "content_insights": {
                "top_words": [word for word, count in word_frequency.most_common(20)],
                "top_emojis": dict(emoji_usage.most_common(10)),
                "top_reactions": dict(reaction_stats.most_common(10)),
                "long_messages": sorted(long_messages, key=lambda x: x["length"], reverse=True)[:5]
            }
        }
        
    def generate_structured_digest(self, analysis: Dict[str, Any]) -> str:
        """Generate a properly formatted weekly digest."""
        if analysis.get("total_messages", 0) == 0:
            return "📭 **Weekly Digest**: No activity detected for the specified timeframe."
            
        overview = analysis["overview"]
        timeframe = analysis["timeframe"]
        
        # Build structured digest
        digest = []
        
        # Header
        digest.append("# 📊 Weekly Discord Community Digest")
        digest.append(f"**Period**: {timeframe['start'][:10]} to {timeframe['end'][:10]} ({timeframe['days']} days)")
        digest.append("")
        
        # Overview
        digest.append("## 🔥 Activity Overview")
        digest.append(f"- **💬 Total Messages**: {overview['total_messages']:,}")
        digest.append(f"- **📺 Active Channels**: {overview['active_channels']}")
        digest.append(f"- **👥 Active Users**: {overview['active_users']}")
        digest.append(f"- **📏 Avg Message Length**: {overview['avg_message_length']} characters")
        digest.append(f"- **🔗 Links Shared**: {overview['total_links_shared']}")
        digest.append(f"- **⚡ Total Reactions**: {overview['total_reactions']}")
        digest.append("")
        
        # Top Channels
        digest.append("## 📺 Most Active Channels")
        for i, (channel, count) in enumerate(list(analysis["top_channels"].items())[:5], 1):
            digest.append(f"{i}. **{channel}**: {count} messages")
        digest.append("")
        
        # Top Contributors
        digest.append("## 👑 Top Contributors")
        for i, (user, count) in enumerate(list(analysis["top_users"].items())[:5], 1):
            digest.append(f"{i}. **{user}**: {count} messages")
        digest.append("")
        
        # Peak Activity
        peak = analysis["peak_activity"]
        digest.append("## ⏰ Peak Activity")
        digest.append(f"- **Busiest Day**: {peak['day']} ({peak['day_messages']} messages)")
        digest.append(f"- **Peak Hour**: {peak['hour']}:00 UTC ({peak['messages']} messages)")
        digest.append("")
        
        # Content Highlights
        content = analysis["content_insights"]
        digest.append("## 🎯 Content Highlights")
        
        if content["top_words"]:
            digest.append("**🔤 Trending Keywords**: " + ", ".join(content["top_words"][:10]))
            
        if content["top_emojis"]:
            emoji_list = [f"{emoji} ({count})" for emoji, count in list(content["top_emojis"].items())[:5]]
            digest.append("**😀 Top Emojis**: " + ", ".join(emoji_list))
            
        if content["top_reactions"]:
            reaction_list = [f"{emoji} ({count})" for emoji, count in list(content["top_reactions"].items())[:5]]
            digest.append("**⚡ Popular Reactions**: " + ", ".join(reaction_list))
            
        digest.append("")
        
        # Notable Messages
        if content["long_messages"]:
            digest.append("## 📝 Notable Long-form Discussions")
            for msg in content["long_messages"][:3]:
                digest.append(f"- **{msg['author']}** in **{msg['channel']}** ({msg['length']} chars)")
                digest.append(f"  _{msg['preview']}_")
            digest.append("")
            
        # Daily Breakdown
        digest.append("## 📅 Daily Activity Breakdown")
        daily = analysis["daily_activity"]
        for day, count in sorted(daily.items()):
            digest.append(f"- **{day}**: {count} messages")
        digest.append("")
        
        # Footer
        digest.append("---")
        digest.append("_Generated by Enhanced Weekly Digest System_")
        
        return "\n".join(digest)
        
    def test_enhanced_vs_current(self) -> Dict[str, Any]:
        """Compare enhanced digest with current agent output."""
        print("🆚 Testing Enhanced vs Current Agent")
        
        start_date, end_date = self.get_week_timeframe()
        
        # Enhanced approach
        print("📊 Generating enhanced digest...")
        analysis = self.analyze_week_data(start_date, end_date)
        enhanced_digest = self.generate_structured_digest(analysis)
        
        # Current agent approach
        print("🤖 Testing current agent...")
        query = "write a weekly digest with the key highlights of the discord community"
        start_time = datetime.now()
        try:
            current_digest = get_agent_answer(query)
            agent_time = (datetime.now() - start_time).total_seconds()
            agent_success = True
        except Exception as e:
            current_digest = f"Error: {str(e)}"
            agent_time = 0
            agent_success = False
            
        return {
            "timeframe": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "data_analysis": analysis,
            "enhanced_digest": enhanced_digest,
            "current_agent": {
                "response": current_digest,
                "success": agent_success,
                "response_time": agent_time
            },
            "comparison": {
                "enhanced_length": len(enhanced_digest),
                "current_length": len(current_digest),
                "enhanced_structure": enhanced_digest.count("#"),
                "current_structure": current_digest.count("#"),
                "enhanced_coverage": {
                    "channels": len(analysis.get("top_channels", {})),
                    "users": len(analysis.get("top_users", {})),
                    "metrics": analysis["overview"]["total_messages"]
                }
            }
        }
        
    def save_comparison_report(self, results: Dict[str, Any], filename: str = "enhanced_digest_comparison.json"):
        """Save detailed comparison report."""
        filepath = f"tests/{filename}"
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Comparison report saved to: {filepath}")
        
    def print_comparison_summary(self, results: Dict[str, Any]):
        """Print human-readable comparison summary."""
        print("\n" + "="*80)
        print("📊 ENHANCED WEEKLY DIGEST COMPARISON")
        print("="*80)
        
        data = results["data_analysis"]
        enhanced = results["enhanced_digest"]
        current = results["current_agent"]
        comparison = results["comparison"]
        
        print(f"📅 Period: {results['timeframe']['start'][:10]} to {results['timeframe']['end'][:10]}")
        print(f"📊 Data Found: {data['overview']['total_messages']} messages, {data['overview']['active_channels']} channels")
        print()
        
        print("🔍 COMPARISON RESULTS:")
        print(f"   Enhanced Length: {comparison['enhanced_length']:,} chars")
        print(f"   Current Length:  {comparison['current_length']:,} chars")
        print(f"   Enhanced Structure: {comparison['enhanced_structure']} sections")
        print(f"   Current Structure:  {comparison['current_structure']} sections")
        print()
        
        print("📈 ENHANCED DIGEST COVERAGE:")
        cov = comparison["enhanced_coverage"]
        print(f"   ✅ Channels Covered: {cov['channels']}")
        print(f"   ✅ Users Mentioned: {cov['users']}")
        print(f"   ✅ Total Messages Analyzed: {cov['metrics']:,}")
        print()
        
        print("🤖 CURRENT AGENT STATUS:")
        print(f"   Success: {'✅' if current['success'] else '❌'}")
        print(f"   Response Time: {current['response_time']:.2f}s")
        print()
        
        # Show sample of enhanced digest
        print("📄 ENHANCED DIGEST PREVIEW:")
        preview_lines = enhanced.split('\n')[:15]
        for line in preview_lines:
            print(f"   {line}")
        print("   ...")
        print()
        
        print("🎯 RECOMMENDATION:")
        if comparison['enhanced_length'] > comparison['current_length'] * 1.5:
            print("   ✅ Enhanced digest provides significantly more comprehensive coverage")
        if comparison['enhanced_structure'] > comparison['current_structure']:
            print("   ✅ Enhanced digest has better structural organization")
        if cov['channels'] > 5:
            print("   ✅ Enhanced digest covers multiple channels effectively")
            
        print("="*80)

def main():
    """Main execution function."""
    print("🚀 Starting Enhanced Weekly Digest Comparison")
    print("="*50)
    
    try:
        with EnhancedWeeklyDigest() as digest_gen:
            # Run comprehensive comparison
            results = digest_gen.test_enhanced_vs_current()
            
            # Save and display results
            digest_gen.save_comparison_report(results)
            digest_gen.print_comparison_summary(results)
            
            return 0
            
    except Exception as e:
        print(f"❌ Error during comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
