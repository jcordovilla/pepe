#!/usr/bin/env python3
"""
Enhanced Community-Focused Preprocessor for Discord Messages

This module implements advanced preprocessing specifically designed for
community-focused RAG agent capabilities including expert identification,
skill mining, conversation threading, and engagement analysis.

Key enhancements:
1. Expert Identification & Skill Mining
2. Conversation Threading & Context Analysis
3. Community Activity & Engagement Metrics
4. Temporal & Event Mining
5. Question/Answer Pattern Detection
6. Resource & Tutorial Classification
"""

import os
import sys
import json
import re
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from db import SessionLocal, Message
from utils.logger import setup_logging

setup_logging()
import logging
log = logging.getLogger(__name__)

@dataclass
class CommunityMetadata:
    """Enhanced metadata structure for community-focused analysis"""
    # Expert identification
    skill_keywords: List[str]
    expertise_indicators: Dict[str, float]  # skill -> confidence score
    question_indicators: bool
    solution_indicators: bool
    help_seeking: bool
    help_providing: bool
    
    # Conversation context
    conversation_thread_id: Optional[str]
    reply_depth: int
    thread_participants: List[str]
    question_resolved: Optional[bool]
    resolution_confidence: float
    
    # Community engagement
    reaction_sentiment: Dict[str, int]  # positive/negative/neutral
    engagement_score: float
    influence_score: float
    topic_category: str
    
    # Temporal & events
    event_mentions: List[Dict[str, Any]]
    deadline_indicators: List[str]
    time_sensitive: bool
    
    # Content classification
    content_type: str  # tutorial, question, announcement, discussion, etc.
    resource_quality: float  # 0-1 score for resource posts
    code_snippets: List[str]
    tutorial_steps: List[str]

class CommunityPreprocessor:
    """Advanced preprocessor for community-focused Discord analysis"""
    
    def __init__(self):
        self.skill_patterns = self._load_skill_patterns()
        self.question_patterns = self._load_question_patterns()
        self.solution_patterns = self._load_solution_patterns()
        self.event_patterns = self._load_event_patterns()
        self.positive_reactions = {'👍', '❤️', '🎉', '💯', '🔥', '✅', '⭐'}
        self.negative_reactions = {'👎', '😞', '❌', '⚠️', '🚨'}
        
    def _load_skill_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for identifying technical skills and expertise"""
        return {
            'programming_languages': [
                r'\bpython\b', r'\bjavascript\b', r'\bjs\b', r'\btypescript\b', r'\bts\b',
                r'\bjava\b', r'\bc\+\+\b', r'\bc#\b', r'\bruby\b', r'\bphp\b', r'\bgo\b',
                r'\brust\b', r'\bkotlin\b', r'\bswift\b', r'\bdart\b', r'\bscala\b'
            ],
            'frameworks': [
                r'\breact\b', r'\bvue\b', r'\bangular\b', r'\bdjango\b', r'\bflask\b',
                r'\bspring\b', r'\bexpress\b', r'\bnext\.?js\b', r'\bnuxt\b', r'\bsvelte\b'
            ],
            'ai_ml': [
                r'\bmachine learning\b', r'\bml\b', r'\bai\b', r'\bneural network\b',
                r'\btensorflow\b', r'\bpytorch\b', r'\btransformer\b', r'\bgpt\b', r'\bllm\b',
                r'\bembedding\b', r'\bfaiss\b', r'\bvector\b', r'\brag\b'
            ],
            'devops': [
                r'\bdocker\b', r'\bkubernetes\b', r'\bk8s\b', r'\baws\b', r'\bazure\b',
                r'\bgcp\b', r'\bci/cd\b', r'\bjenkins\b', r'\bgithub actions\b', r'\bterraform\b'
            ],
            'databases': [
                r'\bmysql\b', r'\bpostgresql\b', r'\bmongodb\b', r'\bredis\b', r'\bsqlite\b',
                r'\belasticsearch\b', r'\bneo4j\b', r'\bcassandra\b'
            ],
            'discord': [
                r'\bdiscord\.py\b', r'\bdiscord bot\b', r'\bslash command\b', r'\bembed\b',
                r'\boauth\b', r'\brate limit\b', r'\bapi\b', r'\bwebhook\b'
            ]
        }
    
    def _load_question_patterns(self) -> List[str]:
        """Patterns indicating help-seeking behavior"""
        return [
            r'\bhow do i\b', r'\bhow to\b', r'\bcan someone\b', r'\bneed help\b',
            r'\bhelp me\b', r'\bstuck with\b', r'\berror\b', r'\bissue\b', r'\bproblem\b',
            r'\bdoes anyone know\b', r'\bhas anyone\b', r'\bwhat\'?s the best\b',
            r'\brecommend\b', r'\bsuggestion\b', r'\badvice\b', r'\?\s*$'
        ]
    
    def _load_solution_patterns(self) -> List[str]:
        """Patterns indicating help-providing behavior"""
        return [
            r'\byou can\b', r'\btry this\b', r'\bhere\'?s how\b', r'\bsolution\b',
            r'\bworked for me\b', r'\bi fixed\b', r'\bsteps:\b', r'\btutorial\b',
            r'\bguide\b', r'\bexample\b', r'\bhere\'?s the code\b', r'\bfixed it\b'
        ]
    
    def _load_event_patterns(self) -> List[str]:
        """Patterns for extracting event mentions and deadlines"""
        return [
            r'\btomorrow\b', r'\bnext week\b', r'\bnext month\b', r'\bthis week\b',
            r'\bby friday\b', r'\bdeadline\b', r'\bdue date\b', r'\bevent\b',
            r'\bmeeting\b', r'\bworkshop\b', r'\bwebinar\b', r'\bhackathon\b'
        ]
    
    def extract_skills(self, content: str) -> Tuple[List[str], Dict[str, float]]:
        """Extract technical skills and calculate expertise confidence"""
        content_lower = content.lower()
        skills = []
        expertise_scores = {}
        
        for category, patterns in self.skill_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content_lower, re.IGNORECASE)
                if matches:
                    skill = pattern.replace(r'\b', '').replace(r'\.?', '.')
                    skills.append(skill)
                    
                    # Calculate expertise confidence based on context
                    confidence = self._calculate_expertise_confidence(content_lower, skill)
                    expertise_scores[skill] = confidence
        
        return list(set(skills)), expertise_scores
    
    def _calculate_expertise_confidence(self, content: str, skill: str) -> float:
        """Calculate confidence that user has expertise in this skill"""
        confidence = 0.5  # Base confidence
        
        # Boost for solution-providing language
        if any(re.search(pattern, content) for pattern in self.solution_patterns):
            confidence += 0.3
        
        # Boost for detailed explanations
        if len(content) > 200:
            confidence += 0.1
        
        # Boost for code examples
        if '```' in content or '`' in content:
            confidence += 0.2
        
        # Reduce for question-asking language
        if any(re.search(pattern, content) for pattern in self.question_patterns):
            confidence -= 0.2
        
        return min(1.0, max(0.0, confidence))
    
    def analyze_question_answer_patterns(self, content: str) -> Dict[str, bool]:
        """Detect if message is asking or providing help"""
        content_lower = content.lower()
        
        is_question = any(re.search(pattern, content_lower) for pattern in self.question_patterns)
        is_solution = any(re.search(pattern, content_lower) for pattern in self.solution_patterns)
        
        # More sophisticated detection
        question_score = 0
        solution_score = 0
        
        # Question indicators
        if '?' in content:
            question_score += 0.3
        if any(word in content_lower for word in ['help', 'stuck', 'error', 'issue', 'problem']):
            question_score += 0.2
        if content.startswith(('How', 'What', 'Why', 'Where', 'When', 'Can', 'Does')):
            question_score += 0.3
        
        # Solution indicators
        if any(word in content_lower for word in ['solved', 'fixed', 'solution', 'answer']):
            solution_score += 0.3
        if '```' in content:  # Code blocks often indicate solutions
            solution_score += 0.2
        if len(content) > 100:  # Longer messages more likely to be explanatory
            solution_score += 0.1
        
        return {
            'question_indicators': is_question or question_score > 0.4,
            'solution_indicators': is_solution or solution_score > 0.4,
            'help_seeking': question_score > 0.3,
            'help_providing': solution_score > 0.3
        }
    
    def extract_code_snippets(self, content: str) -> List[str]:
        """Extract code blocks and inline code"""
        snippets = []
        
        # Extract code blocks (```...```)
        code_blocks = re.findall(r'```(?:[\w]*\n)?(.*?)```', content, re.DOTALL)
        snippets.extend([block.strip() for block in code_blocks if block.strip()])
        
        # Extract inline code (`...`)
        inline_code = re.findall(r'`([^`]+)`', content)
        snippets.extend([code.strip() for code in inline_code if len(code.strip()) > 5])
        
        return snippets
    
    def classify_content_type(self, content: str, attachments: List[Dict], embeds: List[Dict]) -> str:
        """Classify the type of content"""
        content_lower = content.lower()
        
        # Tutorial indicators
        if any(word in content_lower for word in ['tutorial', 'guide', 'step', 'how to']):
            return 'tutorial'
        
        # Question
        if '?' in content or any(word in content_lower for word in ['help', 'how', 'what', 'why']):
            return 'question'
        
        # Announcement
        if any(word in content_lower for word in ['announce', 'reminder', 'notice', 'update']):
            return 'announcement'
        
        # Resource sharing
        if attachments or embeds or any(word in content_lower for word in ['link', 'resource', 'article']):
            return 'resource'
        
        # Code sharing
        if '```' in content:
            return 'code'
        
        return 'discussion'
    
    def analyze_engagement(self, reactions: List[Dict], mention_count: int, content_length: int) -> Dict[str, float]:
        """Analyze engagement metrics"""
        reaction_sentiment = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_reactions = 0
        
        for reaction in reactions:
            emoji = reaction.get('emoji', '')
            count = reaction.get('count', 0)
            total_reactions += count
            
            if emoji in self.positive_reactions:
                reaction_sentiment['positive'] += count
            elif emoji in self.negative_reactions:
                reaction_sentiment['negative'] += count
            else:
                reaction_sentiment['neutral'] += count
        
        # Calculate engagement score
        engagement_score = (
            total_reactions * 0.5 +
            mention_count * 0.3 +
            min(content_length / 100, 5) * 0.2
        )
        
        # Calculate influence score (how much this message might be referenced)
        influence_score = (
            reaction_sentiment['positive'] * 0.6 +
            reaction_sentiment['neutral'] * 0.2 +
            content_length / 200 * 0.2
        )
        
        return {
            'reaction_sentiment': reaction_sentiment,
            'engagement_score': min(10.0, engagement_score),
            'influence_score': min(10.0, influence_score)
        }
    
    def extract_events_and_deadlines(self, content: str) -> Tuple[List[Dict], List[str], bool]:
        """Extract event mentions and deadline indicators"""
        content_lower = content.lower()
        events = []
        deadlines = []
        time_sensitive = False
        
        # Find event patterns
        for pattern in self.event_patterns:
            matches = re.findall(pattern, content_lower)
            if matches:
                events.extend([{
                    'type': 'event_mention',
                    'text': match,
                    'pattern': pattern
                } for match in matches])
        
        # Find specific deadline patterns
        deadline_patterns = [
            r'by (\w+)', r'due (\w+)', r'deadline (\w+)',
            r'(\d+)/(\d+)', r'(\w+) (\d+)'
        ]
        
        for pattern in deadline_patterns:
            matches = re.findall(pattern, content_lower)
            if matches:
                deadlines.extend([str(match) for match in matches])
                time_sensitive = True
        
        # Check for urgency indicators
        urgency_words = ['urgent', 'asap', 'immediately', 'quickly', 'soon', 'today', 'tomorrow']
        if any(word in content_lower for word in urgency_words):
            time_sensitive = True
        
        return events, deadlines, time_sensitive
    
    def process_message(self, message: Message) -> CommunityMetadata:
        """Process a single message and extract community-focused metadata"""
        content = message.content or ""
        attachments = message.attachments or []
        embeds = message.embeds or []
        reactions = message.reactions or []
        
        # Extract skills and expertise
        skills, expertise_scores = self.extract_skills(content)
        
        # Analyze question/answer patterns
        qa_analysis = self.analyze_question_answer_patterns(content)
        
        # Extract code snippets
        code_snippets = self.extract_code_snippets(content)
        
        # Classify content type
        content_type = self.classify_content_type(content, attachments, embeds)
        
        # Analyze engagement
        engagement_data = self.analyze_engagement(
            reactions, 
            len(message.mention_ids or []), 
            len(content)
        )
        
        # Extract events and deadlines
        events, deadlines, time_sensitive = self.extract_events_and_deadlines(content)
        
        # Determine conversation threading (simplified for now)
        reply_depth = 1 if message.reference else 0
        thread_id = str(message.channel_id)  # Basic grouping by channel
        
        # Calculate resource quality for resource-type messages
        resource_quality = 0.5
        if content_type == 'resource':
            resource_quality = min(1.0, (
                len(attachments) * 0.2 +
                len(embeds) * 0.3 +
                engagement_data['engagement_score'] / 10 * 0.5
            ))
        
        return CommunityMetadata(
            skill_keywords=skills,
            expertise_indicators=expertise_scores,
            question_indicators=qa_analysis['question_indicators'],
            solution_indicators=qa_analysis['solution_indicators'],
            help_seeking=qa_analysis['help_seeking'],
            help_providing=qa_analysis['help_providing'],
            
            conversation_thread_id=thread_id,
            reply_depth=reply_depth,
            thread_participants=[message.author.get('username', 'unknown')],
            question_resolved=None,  # Would need thread analysis
            resolution_confidence=0.0,
            
            reaction_sentiment=engagement_data['reaction_sentiment'],
            engagement_score=engagement_data['engagement_score'],
            influence_score=engagement_data['influence_score'],
            topic_category=self._categorize_topic(skills, content),
            
            event_mentions=events,
            deadline_indicators=deadlines,
            time_sensitive=time_sensitive,
            
            content_type=content_type,
            resource_quality=resource_quality,
            code_snippets=code_snippets,
            tutorial_steps=self._extract_tutorial_steps(content)
        )
    
    def _categorize_topic(self, skills: List[str], content: str) -> str:
        """Categorize the main topic of the message"""
        content_lower = content.lower()
        
        # AI/ML category
        ai_keywords = ['ai', 'ml', 'machine learning', 'neural', 'gpt', 'llm', 'embedding']
        if any(skill in ai_keywords for skill in skills) or any(kw in content_lower for kw in ai_keywords):
            return 'ai_ml'
        
        # Web development
        web_keywords = ['react', 'vue', 'angular', 'javascript', 'css', 'html', 'frontend', 'backend']
        if any(skill in web_keywords for skill in skills) or any(kw in content_lower for kw in web_keywords):
            return 'web_development'
        
        # DevOps
        devops_keywords = ['docker', 'kubernetes', 'aws', 'deployment', 'ci/cd']
        if any(skill in devops_keywords for skill in skills) or any(kw in content_lower for kw in devops_keywords):
            return 'devops'
        
        # Discord development
        if 'discord' in skills or 'discord' in content_lower:
            return 'discord_development'
        
        # General programming
        if skills:
            return 'programming'
        
        # Community/social
        community_keywords = ['event', 'meetup', 'introduction', 'welcome', 'buddy', 'group']
        if any(kw in content_lower for kw in community_keywords):
            return 'community'
        
        return 'general'
    
    def _extract_tutorial_steps(self, content: str) -> List[str]:
        """Extract step-by-step instructions from tutorial content"""
        steps = []
        
        # Look for numbered steps
        step_patterns = [
            r'(\d+)\.\s*(.+)',
            r'Step \d+[:\-]\s*(.+)',
            r'^\s*-\s*(.+)',
            r'^\s*\*\s*(.+)'
        ]
        
        lines = content.split('\n')
        for line in lines:
            for pattern in step_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    if len(match.groups()) > 1:
                        steps.append(match.group(2).strip())
                    else:
                        steps.append(match.group(1).strip())
                    break
        
        return steps[:10]  # Limit to first 10 steps

def main():
    """Test the community preprocessor"""
    processor = CommunityPreprocessor()
    
    # Test with a sample from the database
    session = SessionLocal()
    try:
        # Get a few sample messages
        messages = session.query(Message).limit(5).all()
        
        for msg in messages:
            print(f"\n{'='*50}")
            print(f"Message ID: {msg.message_id}")
            print(f"Author: {msg.author.get('username', 'unknown')}")
            print(f"Content: {msg.content[:100]}...")
            
            metadata = processor.process_message(msg)
            print(f"Extracted metadata:")
            print(json.dumps(asdict(metadata), indent=2, default=str))
            
    finally:
        session.close()

if __name__ == "__main__":
    main()
