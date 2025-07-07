"""
Enhanced Memory Management

Provides advanced conversation memory with intelligent summarization,
context awareness, and user preference learning.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import asyncio
import json
from collections import defaultdict

from ..services.llm_client import get_llm_client

logger = logging.getLogger(__name__)


class EnhancedConversationMemory:
    """
    Advanced conversation memory with intelligent features:
    - Smart summarization using unified LLM
    - User preference learning
    - Context-aware retrieval
    - Semantic clustering of conversations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_client = get_llm_client()
        
        # Memory configuration
        self.max_active_memory = config.get("max_active_memory", 50)
        self.summary_threshold = config.get("summary_threshold", 20)
        self.context_window_hours = config.get("context_window_hours", 24)
        self.preference_learning = config.get("enable_preference_learning", True)
        
        # User profiles and preferences
        self.user_profiles = {}
        self.conversation_clusters = defaultdict(list)
        
        logger.info("Enhanced memory system initialized")
    
    async def get_contextual_history(
        self, 
        user_id: str, 
        current_query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get contextually relevant conversation history.
        
        Args:
            user_id: User identifier
            current_query: Current user query for context
            limit: Maximum number of items to return
            
        Returns:
            Contextually relevant conversation history
        """
        try:
            # Get recent conversations
            recent_history = await self._get_recent_history(user_id, limit * 2)
            
            if not recent_history:
                return []
            
            # Score conversations by relevance to current query
            scored_conversations = []
            for conv in recent_history:
                relevance_score = await self._calculate_relevance(
                    current_query, 
                    conv.get("query", ""),
                    conv.get("response", "")
                )
                scored_conversations.append((conv, relevance_score))
            
            # Sort by relevance and return top results
            scored_conversations.sort(key=lambda x: x[1], reverse=True)
            return [conv for conv, score in scored_conversations[:limit]]
            
        except Exception as e:
            logger.error(f"Error getting contextual history: {e}")
            return []
    
    async def learn_user_preferences(
        self, 
        user_id: str, 
        query: str, 
        response: str,
        feedback: Optional[Dict[str, Any]] = None
    ):
        """
        Learn user preferences from interactions.
        
        Args:
            user_id: User identifier
            query: User query
            response: System response
            feedback: Optional user feedback
        """
        if not self.preference_learning:
            return
        
        try:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {
                    "query_patterns": defaultdict(int),
                    "preferred_response_length": "medium",
                    "topics_of_interest": defaultdict(int),
                    "interaction_style": "formal",
                    "created_at": datetime.utcnow().isoformat()
                }
            
            profile = self.user_profiles[user_id]
            
            # Learn query patterns
            query_type = await self._classify_query_type(query)
            profile["query_patterns"][query_type] += 1
            
            # Learn response length preference (if feedback provided)
            if feedback and "length_preference" in feedback:
                profile["preferred_response_length"] = feedback["length_preference"]
            
            # Extract topics of interest
            topics = await self._extract_topics(query + " " + response)
            for topic in topics:
                profile["topics_of_interest"][topic] += 1
            
            logger.debug(f"Updated preferences for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error learning user preferences: {e}")
    
    async def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive user context including preferences and patterns.
        
        Args:
            user_id: User identifier
            
        Returns:
            User context information
        """
        try:
            base_context = {
                "user_id": user_id,
                "last_interaction": None,
                "interaction_count": 0,
                "preferences": {},
                "common_topics": [],
                "query_patterns": {}
            }
            
            if user_id in self.user_profiles:
                profile = self.user_profiles[user_id]
                
                # Get top topics of interest
                top_topics = sorted(
                    profile["topics_of_interest"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                # Get common query patterns
                top_patterns = sorted(
                    profile["query_patterns"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                base_context.update({
                    "preferences": {
                        "response_length": profile.get("preferred_response_length", "medium"),
                        "interaction_style": profile.get("interaction_style", "formal")
                    },
                    "common_topics": [topic for topic, count in top_topics],
                    "query_patterns": dict(top_patterns),
                    "profile_created": profile.get("created_at")
                })
            
            return base_context
            
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return {"user_id": user_id, "error": str(e)}
    
    async def smart_summarize_conversation(
        self, 
        conversations: List[Dict[str, Any]]
    ) -> str:
        """
        Create intelligent conversation summary using LLM.
        
        Args:
            conversations: List of conversation interactions
            
        Returns:
            Intelligent summary of conversations
        """
        try:
            if not conversations:
                return ""
            
            # Prepare conversation text
            conversation_text = ""
            for conv in conversations:
                query = conv.get("query", "")
                response = conv.get("response", "")
                timestamp = conv.get("timestamp", "")
                
                conversation_text += f"[{timestamp}] Q: {query}\nA: {response}\n\n"
            
            # Generate summary using LLM
            summary_prompt = f"""
            Summarize the following conversation history into key points and themes.
            Focus on:
            1. Main topics discussed
            2. User preferences and patterns
            3. Important context for future interactions
            4. Any ongoing discussions or follow-ups needed
            
            Conversation History:
            {conversation_text}
            
            Provide a concise but comprehensive summary:
            """
            
            summary = await self.llm_client.generate(
                prompt=summary_prompt,
                system_prompt="You are an expert at conversation summarization.",
                max_tokens=500,
                temperature=0.1
            )
            return summary or "No significant conversation patterns identified."
            
        except Exception as e:
            logger.error(f"Error in smart summarization: {e}")
            return "Error generating conversation summary."
    
    async def _calculate_relevance(
        self, 
        current_query: str, 
        historical_query: str, 
        historical_response: str
    ) -> float:
        """Calculate relevance score between current and historical interactions."""
        try:
            # Simple keyword-based relevance (could be enhanced with embeddings)
            current_words = set(current_query.lower().split())
            historical_words = set((historical_query + " " + historical_response).lower().split())
            
            if not current_words or not historical_words:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(current_words.intersection(historical_words))
            union = len(current_words.union(historical_words))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 0.0
    
    async def _classify_query_type(self, query: str) -> str:
        """Classify query into type categories."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["search", "find", "look for"]):
            return "search"
        elif any(word in query_lower for word in ["digest", "summary", "summarize"]):
            return "digest"
        elif any(word in query_lower for word in ["analyze", "explain", "what is"]):
            return "analysis"
        elif any(word in query_lower for word in ["help", "how to", "guide"]):
            return "help"
        else:
            return "general"
    
    async def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text using simple keyword extraction."""
        # Simple topic extraction (could be enhanced with NLP)
        technical_keywords = [
            "ai", "machine learning", "python", "discord", "bot", "api",
            "database", "search", "vector", "embedding", "llm", "openai"
        ]
        
        text_lower = text.lower()
        found_topics = [keyword for keyword in technical_keywords if keyword in text_lower]
        
        return found_topics
    
    async def _get_recent_history(self, user_id: str, limit: int) -> List[Dict[str, Any]]:
        """Get recent conversation history (placeholder - integrate with actual storage)."""
        # This would integrate with the actual conversation storage
        return [] 