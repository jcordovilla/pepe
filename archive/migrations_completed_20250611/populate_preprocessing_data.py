#!/usr/bin/env python3
"""
Preprocessing Population Script

This script populates the newly added preprocessing fields with AI-generated content
to enable enhanced weekly digest capabilities.

Processes:
1. Enhanced content generation with context
2. Topic and keyword extraction
3. Intent and sentiment analysis
4. Engagement scoring
5. Content type classification
6. Technology mention detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import Counter

from db.db import get_db_session, Message
from core.ai_client import get_ai_client
from sqlalchemy import func, and_
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessagePreprocessor:
    """Enhanced message preprocessor for weekly digest capabilities."""
    
    def __init__(self):
        self.ai_client = get_ai_client()
        self.tech_keywords = [
            "python", "javascript", "typescript", "react", "vue", "angular", "node",
            "docker", "kubernetes", "aws", "azure", "gcp", "mysql", "postgresql", 
            "mongodb", "redis", "elasticsearch", "api", "rest", "graphql", "sql",
            "machine learning", "ai", "neural network", "tensorflow", "pytorch",
            "github", "git", "ci/cd", "jenkins", "terraform", "ansible", "linux",
            "windows", "macos", "ios", "android", "swift", "kotlin", "java", "c++",
            "rust", "go", "php", "ruby", "scala", "blockchain", "cryptocurrency"
        ]
        
    def extract_topics_and_keywords(self, content: str, channel_name: str = "") -> Dict[str, List[str]]:
        """Extract topics and keywords from message content using AI."""
        if not content or len(content.strip()) < 10:
            return {"topics": [], "keywords": []}
            
        try:
            prompt = f"""Analyze this Discord message and extract key information:

Message: "{content}"
Channel: {channel_name}

Extract:
1. Main topics/themes (max 3)
2. Important keywords/entities (max 5)

Return as JSON: {{"topics": ["topic1", "topic2"], "keywords": ["keyword1", "keyword2"]}}

Focus on:
- Technical discussions
- Project mentions  
- Problem-solving topics
- Community activities
- Learning resources"""

            messages = [{"role": "user", "content": prompt}]
            response = self.ai_client.chat_completion(messages, max_tokens=200)
            
            # Try to parse JSON response
            try:
                result = json.loads(response.strip())
                if isinstance(result, dict) and "topics" in result and "keywords" in result:
                    return {
                        "topics": result.get("topics", [])[:3],
                        "keywords": result.get("keywords", [])[:5]
                    }
            except json.JSONDecodeError:
                pass
                
            # Fallback: extract manually
            return self._extract_keywords_manually(content)
            
        except Exception as e:
            logger.warning(f"AI extraction failed: {e}")
            return self._extract_keywords_manually(content)
    
    def _extract_keywords_manually(self, content: str) -> Dict[str, List[str]]:
        """Fallback manual keyword extraction."""
        content_lower = content.lower()
        
        # Extract technical terms
        found_tech = [tech for tech in self.tech_keywords if tech in content_lower]
        
        # Extract common patterns
        keywords = []
        
        # URLs/links
        if "http" in content_lower or "www." in content_lower:
            keywords.append("link-sharing")
            
        # Questions
        if any(q in content_lower for q in ["how", "what", "why", "when", "where"]):
            keywords.append("question")
            
        # Code patterns
        if any(pattern in content for pattern in ["```", "`", "function", "class", "import"]):
            keywords.append("code")
            
        # Combine with tech terms
        all_keywords = found_tech + keywords
        
        # Simple topic extraction
        topics = []
        if found_tech:
            topics.append("technology")
        if "question" in keywords:
            topics.append("help-seeking")
        if "code" in keywords:
            topics.append("programming")
            
        return {
            "topics": topics[:3],
            "keywords": all_keywords[:5]
        }
    
    def classify_intent(self, content: str, has_question_mark: bool = False) -> str:
        """Classify message intent."""
        content_lower = content.lower().strip()
        
        # Question patterns
        question_patterns = [
            r'\b(how|what|why|when|where|which|who|can|could|would|should|is|are|do|does|did)\b',
            r'\?'
        ]
        
        if has_question_mark or any(re.search(pattern, content_lower) for pattern in question_patterns):
            return "question"
            
        # Sharing patterns
        if any(word in content_lower for word in ["check out", "found", "here's", "link", "resource"]):
            return "sharing"
            
        # Problem solving
        if any(word in content_lower for word in ["error", "issue", "problem", "bug", "fix", "help"]):
            return "help-seeking"
            
        # Announcement
        if any(word in content_lower for word in ["announcing", "released", "update", "new version"]):
            return "announcement"
            
        # Discussion
        if len(content.split()) > 20:
            return "discussion"
            
        return "general"
    
    def analyze_sentiment(self, content: str) -> str:
        """Simple sentiment analysis."""
        content_lower = content.lower()
        
        positive_words = [
            "great", "awesome", "excellent", "good", "nice", "helpful", "thanks", 
            "amazing", "perfect", "love", "appreciate", "fantastic", "wonderful"
        ]
        
        negative_words = [
            "bad", "terrible", "awful", "hate", "annoying", "frustrated", "angry",
            "stupid", "worst", "sucks", "broken", "useless", "disappointed"
        ]
        
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def calculate_engagement_score(self, message: Message) -> Dict[str, Any]:
        """Calculate engagement metrics for a message."""
        try:
            reactions = message.reactions if isinstance(message.reactions, list) else []
            total_reactions = sum(r.get("count", 0) for r in reactions if isinstance(r, dict))
            
            # Length factor
            content_length = len(message.content or "")
            length_score = min(content_length / 200, 2.0)  # Normalize to max 2.0
            
            # Time factor (newer messages get slight boost)
            now = datetime.now()
            age_days = (now - message.timestamp).days if message.timestamp else 365
            recency_score = max(0, 1.0 - (age_days / 30))  # Decay over 30 days
            
            engagement_score = total_reactions + length_score + recency_score
            
            return {
                "total_reactions": total_reactions,
                "reaction_types": len(reactions),
                "content_length": content_length,
                "recency_score": round(recency_score, 2),
                "overall_score": round(engagement_score, 2)
            }
            
        except Exception as e:
            logger.warning(f"Engagement calculation failed: {e}")
            return {"overall_score": 0}
    
    def classify_content_type(self, message: Message) -> str:
        """Classify the type of content in the message."""
        content = (message.content or "").lower()
        
        # Check for attachments
        if message.attachments:
            return "media"
            
        # Check for embeds
        if message.embeds:
            return "rich-content"
            
        # Check for code
        if "```" in message.content or "`" in message.content:
            return "code"
            
        # Check for links
        if "http" in content or "www." in content:
            return "link"
            
        # Check for questions
        if "?" in content or any(q in content for q in ["how", "what", "why"]):
            return "question"
            
        # Check length
        if len(content.split()) > 50:
            return "long-form"
        elif len(content.split()) < 5:
            return "short"
        else:
            return "discussion"
    
    def detect_mentioned_technologies(self, content: str) -> List[str]:
        """Detect technology mentions in the message."""
        content_lower = content.lower()
        mentioned = []
        
        for tech in self.tech_keywords:
            if tech in content_lower:
                mentioned.append(tech)
                
        return mentioned[:5]  # Limit to top 5
    
    def process_message(self, message: Message) -> Dict[str, Any]:
        """Process a single message and generate all preprocessing data."""
        try:
            content = message.content or ""
            channel_name = message.channel_name or ""
            
            # Extract topics and keywords
            topics_keywords = self.extract_topics_and_keywords(content, channel_name)
            
            # Classify intent
            intent = self.classify_intent(content, "?" in content)
            
            # Analyze sentiment
            sentiment = self.analyze_sentiment(content)
            
            # Calculate engagement
            engagement = self.calculate_engagement_score(message)
            
            # Classify content type
            content_type = self.classify_content_type(message)
            
            # Detect technologies
            technologies = self.detect_mentioned_technologies(content)
            
            # Generate enhanced content (simplified)
            enhanced_content = f"[{channel_name}] {content}"
            if technologies:
                enhanced_content += f" [Technologies: {', '.join(technologies)}]"
            
            return {
                "enhanced_content": enhanced_content,
                "topics": topics_keywords["topics"],
                "keywords": topics_keywords["keywords"],
                "intent": intent,
                "sentiment": sentiment,
                "engagement_score": engagement,
                "content_type": content_type,
                "mentioned_technologies": technologies
            }
            
        except Exception as e:
            logger.error(f"Message processing failed: {e}")
            return {
                "enhanced_content": message.content,
                "topics": [],
                "keywords": [],
                "intent": "general",
                "sentiment": "neutral",
                "engagement_score": {"overall_score": 0},
                "content_type": "general",
                "mentioned_technologies": []
            }
    
    def process_recent_messages(self, days: int = 30, batch_size: int = 100) -> int:
        """Process recent messages in batches."""
        cutoff_date = datetime.now() - timedelta(days=days)
        processed_count = 0
        
        with get_db_session() as session:
            # Get messages that need processing
            messages = session.query(Message).filter(
                and_(
                    Message.timestamp >= cutoff_date,
                    Message.enhanced_content.is_(None)  # Only unprocessed messages
                )
            ).limit(batch_size * 10).all()  # Get more than we need
            
            logger.info(f"Found {len(messages)} unprocessed recent messages")
            
            for i, message in enumerate(messages):
                if i >= batch_size:
                    break
                    
                try:
                    # Process message
                    processed_data = self.process_message(message)
                    
                    # Update database
                    for field, value in processed_data.items():
                        if hasattr(message, field):
                            if field in ["topics", "keywords", "engagement_score", "mentioned_technologies"]:
                                setattr(message, field, json.dumps(value) if value else None)
                            else:
                                setattr(message, field, value)
                    
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        logger.info(f"Processed {processed_count} messages...")
                        session.commit()
                        
                except Exception as e:
                    logger.error(f"Failed to process message {message.id}: {e}")
                    
            # Final commit
            session.commit()
            
        return processed_count

def main():
    """Main preprocessing execution."""
    print("🚀 Starting Message Preprocessing for Enhanced Weekly Digest")
    print("=" * 70)
    
    try:
        preprocessor = MessagePreprocessor()
        
        # Process recent messages (last 30 days)
        print("🔧 Processing recent messages...")
        processed_count = preprocessor.process_recent_messages(days=30, batch_size=200)
        
        print(f"\n✅ Preprocessing completed!")
        print(f"📊 Processed {processed_count} messages")
        print("🎯 Enhanced weekly digest capability is now available!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
