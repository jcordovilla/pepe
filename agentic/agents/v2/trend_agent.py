"""
Trend Analysis Agent

Analyzes temporal trends and patterns in Discord conversations.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import re

from ..base_agent import BaseAgent, AgentRole, AgentState
from ...services.llm_client import UnifiedLLMClient
from ...vectorstore.persistent_store import PersistentVectorStore
from ...utils.k_value_calculator import KValueCalculator

logger = logging.getLogger(__name__)


class TrendAgent(BaseAgent):
    """
    Agent for analyzing trends and patterns in Discord conversations.
    
    Features:
    - Topic trend analysis
    - Temporal pattern detection
    - User engagement trends
    - Content evolution tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.ANALYZER, config)
        self.llm_client = UnifiedLLMClient(config.get("llm", {}))
        self.vector_store = PersistentVectorStore(config.get("vector_config", {}))
        
        # Initialize dynamic k-value calculator
        self.k_calculator = KValueCalculator(config)
        
        # Default search parameters (legacy, kept for backward compatibility)
        self.default_k = config.get("default_k", 5)
        
        # Analysis configuration
        self.min_cluster_size = config.get("min_cluster_size", 3)
        self.max_clusters = config.get("max_clusters", 10)
        
        logger.info("TrendAgent initialized with dynamic k-value calculator")
    
    def signature(self) -> Dict[str, Any]:
        """Return agent signature for registration."""
        return {
            "role": "trend",
            "description": "Detects topics and trends using clustering analysis",
            "input_schema": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "Start date (ISO format)"},
                    "end": {"type": "string", "description": "End date (ISO format)"},
                    "k": {"type": "integer", "description": "Number of topics to detect"}
                },
                "required": ["start", "end"]
            },
            "output_schema": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of detected topics"
            }
        }
    
    async def run(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Detect trends and topics for the given time period.
        
        Args:
            start: Start date
            end: End date
            k: Number of topics to detect
            
        Returns:
            List of detected topics
        """
        start_str = kwargs.get("start")
        end_str = kwargs.get("end")
        k = kwargs.get("k", self.default_k)
        
        # Parse dates
        try:
            start_date = datetime.fromisoformat(start_str) if start_str else datetime.now() - timedelta(days=7)
            end_date = datetime.fromisoformat(end_str) if end_str else datetime.now()
        except ValueError as e:
            logger.error(f"Date parsing error: {e}")
            return []
        
        logger.info(f"TrendAgent processing trends from {start_date} to {end_date} (k={k})")
        
        try:
            # Get messages for current period
            current_messages = await self._get_messages_for_period(start_date, end_date)
            
            if len(current_messages) < self.min_cluster_size * k:
                return [{"error": f"Insufficient messages for trend analysis. Need at least {self.min_cluster_size * k} messages."}]
            
            # Get messages for comparison period
            comparison_start = start_date - timedelta(days=self.comparison_period_days)
            comparison_messages = await self._get_messages_for_period(comparison_start, start_date)
            
            # Detect topics in current period
            current_topics = await self._detect_topics(current_messages, k)
            
            # Detect topics in comparison period
            comparison_topics = await self._detect_topics(comparison_messages, k)
            
            # Analyze trend directions
            topics_with_trends = await self._analyze_trends(current_topics, comparison_topics)
            
            return topics_with_trends
            
        except Exception as e:
            logger.error(f"TrendAgent error: {e}")
            return [{"error": f"Failed to detect trends: {str(e)}"}]
    
    async def _get_messages_for_period(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get messages for a specific time period with dynamic k-value calculation."""
        try:
            # Calculate appropriate k value for trend analysis
            query = f"trend analysis from {start_date.date()} to {end_date.date()}"
            k_calculation = self.k_calculator.calculate_k_value(
                query=query,
                query_type="trend_analysis",
                entities=None,
                context=None
            )
            
            # Use calculated k value, but ensure minimum for trend analysis
            calculated_k = k_calculation["k_value"]
            analysis_k = max(calculated_k, 500)  # Ensure sufficient data for trend analysis
            
            logger.info(f"Trend analysis using k={analysis_k} (calculated: {calculated_k})")
            
            # Get messages from vector store
            results = await self.vector_store.filter_search(
                filters={
                    "timestamp": {
                        "$gte": start_date.isoformat(),
                        "$lte": end_date.isoformat()
                    }
                },
                k=analysis_k,  # Use calculated k value
                sort_by="timestamp"
            )
            
            # Extract text content and metadata
            messages = []
            for result in results:
                content = result.get("content", "")
                if len(content.strip()) > 10:  # Filter out very short messages
                    messages.append({
                        "content": content,
                        "metadata": result.get("metadata", {}),
                        "embedding": result.get("embedding"),
                        "reactions": result.get("metadata", {}).get("reactions", [])
                    })
            
            return messages
            
        except Exception as e:
            logger.error(f"Error getting messages for period: {e}")
            return []
    
    async def _detect_topics(self, messages: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Detect topics using clustering on message content."""
        if not messages:
            return []
        
        try:
            # Extract text content
            texts = [msg["content"] for msg in messages]
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Perform clustering
            if len(texts) < k:
                k = max(1, len(texts) // 2)
            
            kmeans = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )
            
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Extract topics from clusters
            topics = []
            feature_names = vectorizer.get_feature_names_out()
            
            for cluster_id in range(k):
                cluster_messages = [msg for i, msg in enumerate(messages) if cluster_labels[i] == cluster_id]
                
                if len(cluster_messages) < self.min_cluster_size:
                    continue
                
                # Get cluster center and extract keywords
                cluster_center = kmeans.cluster_centers_[cluster_id]
                top_keyword_indices = np.argsort(cluster_center)[-10:]  # Top 10 keywords
                keywords = [feature_names[i] for i in top_keyword_indices if cluster_center[i] > 0.01]
                
                # Calculate engagement score
                total_reactions = sum(
                    sum(reaction.get("count", 0) for reaction in msg.get("reactions", []))
                    for msg in cluster_messages
                )
                engagement_score = total_reactions / len(cluster_messages) if cluster_messages else 0
                
                # Get representative messages
                representative_messages = [
                    msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                    for msg in cluster_messages[:3]  # Top 3 representative messages
                ]
                
                # Generate topic name using LLM
                topic_name = await self._generate_topic_name(keywords, representative_messages)
                
                topic = {
                    "id": f"topic_{cluster_id}",
                    "name": topic_name,
                    "keywords": keywords[:5],  # Top 5 keywords
                    "message_count": len(cluster_messages),
                    "engagement_score": engagement_score,
                    "trend_direction": "stable",  # Will be updated later
                    "representative_messages": representative_messages
                }
                
                topics.append(topic)
            
            return topics
            
        except Exception as e:
            logger.error(f"Error detecting topics: {e}")
            return []
    
    async def _generate_topic_name(self, keywords: List[str], messages: List[str]) -> str:
        """Generate a descriptive name for a topic using LLM."""
        if not keywords:
            return "General Discussion"
        
        prompt = f"""Generate a short, descriptive name (2-4 words) for a topic based on these keywords and sample messages:

Keywords: {', '.join(keywords[:5])}
Sample messages:
{chr(10).join(f"- {msg}" for msg in messages[:2])}

Return only the topic name, nothing else."""

        try:
            name = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=20,
                temperature=0.1
            )
            return name.strip().strip('"').strip("'")
        except Exception as e:
            logger.error(f"Error generating topic name: {e}")
            return f"Topic: {', '.join(keywords[:2])}"
    
    async def _analyze_trends(self, current_topics: List[Dict[str, Any]], comparison_topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze trend directions by comparing current and previous periods."""
        # Create keyword-based matching between periods
        for current_topic in current_topics:
            trend_direction = await self._determine_trend_direction(
                current_topic, comparison_topics
            )
            current_topic["trend_direction"] = trend_direction
        
        return current_topics
    
    async def _determine_trend_direction(self, current_topic: Dict[str, Any], comparison_topics: List[Dict[str, Any]]) -> str:
        """Determine if a topic is rising, stable, or declining."""
        current_keywords = set(current_topic["keywords"])
        
        # Find similar topics in comparison period
        similar_topics = []
        for comp_topic in comparison_topics:
            comp_keywords = set(comp_topic["keywords"])
            overlap = len(current_keywords.intersection(comp_keywords))
            
            if overlap >= 2:  # At least 2 keywords overlap
                similar_topics.append({
                    "topic": comp_topic,
                    "overlap": overlap,
                    "message_count": comp_topic["message_count"]
                })
        
        if not similar_topics:
            return "rising"  # New topic
        
        # Compare message counts
        avg_comp_count = sum(t["message_count"] for t in similar_topics) / len(similar_topics)
        current_count = current_topic["message_count"]
        
        if current_count > avg_comp_count * 1.5:
            return "rising"
        elif current_count < avg_comp_count * 0.7:
            return "declining"
        else:
            return "stable"
    
    async def process(self, state: AgentState) -> AgentState:
        """Process state through the trend agent."""
        args = state.get("agent_args", {})
        
        # Extract parameters
        start = args.get("start")
        end = args.get("end")
        k = args.get("k", self.default_k)
        
        # If no dates provided, use defaults
        if not start:
            start = (datetime.now() - timedelta(days=7)).isoformat()
        if not end:
            end = datetime.now().isoformat()
        
        topics = await self.run(start=start, end=end, k=k)
        
        # Update state with trends
        state["trend_result"] = topics
        state["response"] = self._format_trends_response(topics)
        
        return state
    
    def _format_trends_response(self, topics: List[Dict[str, Any]]) -> str:
        """Format trends as a readable response."""
        if not topics:
            return "No trends detected for the specified period."
        
        if "error" in topics[0]:
            return f"Error detecting trends: {topics[0]['error']}"
        
        response = "## 📈 Discord Trends & Topics\n\n"
        
        for topic in topics:
            trend_emoji = {
                "rising": "📈",
                "stable": "➡️",
                "declining": "📉"
            }.get(topic["trend_direction"], "➡️")
            
            response += f"### {trend_emoji} {topic['name']}\n"
            response += f"- **Messages:** {topic['message_count']}\n"
            response += f"- **Engagement:** {topic['engagement_score']:.1f} reactions/msg\n"
            response += f"- **Trend:** {topic['trend_direction'].title()}\n"
            response += f"- **Keywords:** {', '.join(topic['keywords'])}\n\n"
        
        return response
    
    def can_handle(self, task) -> bool:
        """Check if this agent can handle the given task."""
        return (task.task_type == "trend" or 
                "trend" in task.description.lower() or
                "topic" in task.description.lower() or
                "pattern" in task.description.lower()) 