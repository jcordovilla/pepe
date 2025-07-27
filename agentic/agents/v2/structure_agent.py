"""
Structure Agent

Channel structure analysis and recommendations agent.
Analyzes Discord channel organization and suggests improvements.
"""

import logging
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime, timedelta
import networkx as nx
from collections import defaultdict

from ..base_agent import BaseAgent, AgentRole, AgentState
from ...services.llm_client import UnifiedLLMClient
from ...mcp import MCPServer

logger = logging.getLogger(__name__)


class Recommendation(TypedDict):
    """Recommendation structure for channel organization."""
    id: str
    type: str  # "merge", "split", "reorganize"
    title: str
    description: str
    channels_involved: List[str]
    rationale: str
    priority: str  # "high", "medium", "low"
    expected_impact: str


class StructureAgent(BaseAgent):
    """
    Structure agent that analyzes channel organization and provides recommendations.
    
    Input: dict()
    Output: list[Recommendation]
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.ANALYZER, config)
        self.llm_client = UnifiedLLMClient(config.get("llm", {}))
        
        # Use MCP server from service container if available, otherwise create our own
        self.mcp_server = None  # Will be set by service container injection
        
        # Fallback: Initialize MCP server if not injected
        mcp_config = {
            "sqlite": {
                "db_path": "data/discord_messages.db"
            },
            "llm": config.get("llm", {})
        }
        self.mcp_server = MCPServer(mcp_config)
        
        # Structure analysis configuration
        self.min_messages_per_channel = config.get("min_messages_per_channel", 10)
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        self.analysis_period_days = config.get("analysis_period_days", 30)
        
        logger.info("StructureAgent initialized")
    
    def signature(self) -> Dict[str, Any]:
        """Return agent signature for registration."""
        return {
            "role": "structure",
            "description": "Analyzes channel structure and provides organization recommendations",
            "input_schema": {
                "type": "object",
                "properties": {},
                "description": "No input parameters required"
            },
            "output_schema": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of channel organization recommendations"
            }
        }
    
    async def run(self, **kwargs) -> List[Recommendation]:
        """
        Analyze channel structure and generate recommendations.
        
        Args:
            No specific parameters required
            
        Returns:
            List of channel organization recommendations
        """
        logger.info("StructureAgent analyzing channel structure")
        
        try:
            # Get channel statistics
            channel_stats = await self._get_channel_statistics()
            
            if not channel_stats:
                return [{"error": "No channel data available for analysis."}]
            
            # Analyze channel similarities
            channel_similarities = await self._analyze_channel_similarities(channel_stats)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(channel_stats, channel_similarities)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"StructureAgent error: {e}")
            return [{"error": f"Failed to analyze structure: {str(e)}"}]
    
    async def _get_channel_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all channels."""
        try:
            # Get all messages from the last analysis period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.analysis_period_days)
            
            # Get messages from MCP server
            query = f"show me 10000 messages from {start_date.isoformat()} to {end_date.isoformat()} ordered by timestamp"
            results = await self.mcp_server.query_messages(query)
            
            # Group by channel
            channel_data = defaultdict(lambda: {
                "message_count": 0,
                "unique_users": set(),
                "total_reactions": 0,
                "avg_message_length": 0,
                "topics": [],
                "messages": []
            })
            
            for result in results:
                channel_id = result.get("channel_id", "unknown")
                channel_name = result.get("channel_name", "Unknown")
                author_id = result.get("author_id", "unknown")
                content = result.get("content", "")
                
                # Parse reactions from the result
                reactions_data = result.get("reactions", "[]")
                if isinstance(reactions_data, str):
                    import json
                    try:
                        reactions = json.loads(reactions_data)
                    except:
                        reactions = []
                else:
                    reactions = reactions_data
                
                channel_data[channel_id]["message_count"] += 1
                channel_data[channel_id]["unique_users"].add(author_id)
                channel_data[channel_id]["total_reactions"] += sum(r.get("count", 0) for r in reactions)
                channel_data[channel_id]["messages"].append({
                    "content": content,
                    "author": result.get("author_display_name", result.get("author_username", "Unknown")),
                    "reactions": reactions
                })
            
            # Convert sets to counts and calculate averages
            stats = {}
            for channel_id, data in channel_data.items():
                if data["message_count"] >= self.min_messages_per_channel:
                    stats[channel_id] = {
                        "message_count": data["message_count"],
                        "unique_users": len(data["unique_users"]),
                        "total_reactions": data["total_reactions"],
                        "avg_reactions_per_message": data["total_reactions"] / data["message_count"] if data["message_count"] > 0 else 0,
                        "messages": data["messages"][:100]  # Keep top 100 messages for analysis
                    }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting channel statistics: {e}")
            return {}
    
    async def _analyze_channel_similarities(self, channel_stats: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Analyze similarities between channels using content analysis."""
        similarities = {}
        
        try:
            channel_ids = list(channel_stats.keys())
            
            for i, channel_id_1 in enumerate(channel_ids):
                similarities[channel_id_1] = {}
                
                for j, channel_id_2 in enumerate(channel_ids):
                    if i == j:
                        similarities[channel_id_1][channel_id_2] = 1.0
                        continue
                    
                    # Calculate similarity based on content overlap
                    similarity = await self._calculate_channel_similarity(
                        channel_stats[channel_id_1],
                        channel_stats[channel_id_2]
                    )
                    
                    similarities[channel_id_1][channel_id_2] = similarity
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error analyzing channel similarities: {e}")
            return {}
    
    async def _calculate_channel_similarity(self, channel_1: Dict[str, Any], channel_2: Dict[str, Any]) -> float:
        """Calculate similarity between two channels."""
        try:
            # Extract content from both channels
            content_1 = " ".join([msg["content"] for msg in channel_1["messages"]])
            content_2 = " ".join([msg["content"] for msg in channel_2["messages"]])
            
            if not content_1.strip() or not content_2.strip():
                return 0.0
            
            # Use LLM to calculate semantic similarity
            prompt = f"""Calculate the similarity between these two Discord channel contents on a scale of 0.0 to 1.0:

Channel 1 content: "{content_1[:500]}..."
Channel 2 content: "{content_2[:500]}..."

Consider:
- Topic overlap
- Discussion themes
- User engagement patterns
- Content type (questions, announcements, discussions, etc.)

Return only a number between 0.0 and 1.0."""

            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=10,
                temperature=0.1
            )
            
            try:
                similarity = float(response.strip())
                return max(0.0, min(1.0, similarity))  # Clamp between 0 and 1
            except ValueError:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating channel similarity: {e}")
            return 0.0
    
    async def _generate_recommendations(self, channel_stats: Dict[str, Any], similarities: Dict[str, Dict[str, float]]) -> List[Recommendation]:
        """Generate channel organization recommendations."""
        recommendations = []
        
        try:
            # Find channels with high similarity (potential merges)
            merge_candidates = await self._find_merge_candidates(similarities)
            for candidate in merge_candidates:
                recommendations.append(candidate)
            
            # Find channels that might benefit from splitting
            split_candidates = await self._find_split_candidates(channel_stats)
            for candidate in split_candidates:
                recommendations.append(candidate)
            
            # Find reorganization opportunities
            reorganize_candidates = await self._find_reorganize_candidates(channel_stats, similarities)
            for candidate in reorganize_candidates:
                recommendations.append(candidate)
            
            # Sort by priority
            recommendations.sort(key=lambda r: {"high": 3, "medium": 2, "low": 1}.get(r.get("priority", "low"), 1), reverse=True)
            
            return recommendations[:10]  # Return top 10 recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    async def _find_merge_candidates(self, similarities: Dict[str, Dict[str, float]]) -> List[Recommendation]:
        """Find channels that could be merged due to high similarity."""
        merge_candidates = []
        
        try:
            processed_pairs = set()
            
            for channel_1, channel_similarities in similarities.items():
                for channel_2, similarity in channel_similarities.items():
                    if channel_1 == channel_2:
                        continue
                    
                    pair_key = tuple(sorted([channel_1, channel_2]))
                    if pair_key in processed_pairs:
                        continue
                    
                    processed_pairs.add(pair_key)
                    
                    if similarity >= self.similarity_threshold:
                        recommendation = await self._create_merge_recommendation(channel_1, channel_2, similarity)
                        if recommendation:
                            merge_candidates.append(recommendation)
            
            return merge_candidates
            
        except Exception as e:
            logger.error(f"Error finding merge candidates: {e}")
            return []
    
    async def _create_merge_recommendation(self, channel_1: str, channel_2: str, similarity: float) -> Optional[Recommendation]:
        """Create a merge recommendation for two similar channels."""
        try:
            prompt = f"""Create a channel merge recommendation for two similar Discord channels.

Channel 1: {channel_1}
Channel 2: {channel_2}
Similarity Score: {similarity:.2f}

Return a JSON object with:
- title: Short title for the recommendation
- description: Brief description of the merge
- rationale: Why these channels should be merged
- priority: "high", "medium", or "low"
- expected_impact: What impact this merge would have"""

            response = await self.llm_client.generate_json(prompt=prompt)
            
            return Recommendation(
                id=f"merge_{channel_1}_{channel_2}",
                type="merge",
                title=response.get("title", f"Merge {channel_1} and {channel_2}"),
                description=response.get("description", ""),
                channels_involved=[channel_1, channel_2],
                rationale=response.get("rationale", ""),
                priority=response.get("priority", "medium"),
                expected_impact=response.get("expected_impact", "")
            )
            
        except Exception as e:
            logger.error(f"Error creating merge recommendation: {e}")
            return None
    
    async def _find_split_candidates(self, channel_stats: Dict[str, Any]) -> List[Recommendation]:
        """Find channels that might benefit from splitting."""
        split_candidates = []
        
        try:
            for channel_id, stats in channel_stats.items():
                # Look for channels with high message count and diverse topics
                if (stats["message_count"] > 1000 and 
                    stats["unique_users"] > 20):
                    
                    # Analyze topic diversity
                    topic_diversity = await self._analyze_topic_diversity(stats["messages"])
                    
                    if topic_diversity > 0.7:  # High diversity threshold
                        recommendation = await self._create_split_recommendation(channel_id, stats, topic_diversity)
                        if recommendation:
                            split_candidates.append(recommendation)
            
            return split_candidates
            
        except Exception as e:
            logger.error(f"Error finding split candidates: {e}")
            return []
    
    async def _analyze_topic_diversity(self, messages: List[Dict[str, Any]]) -> float:
        """Analyze the diversity of topics in a channel."""
        try:
            if len(messages) < 10:
                return 0.0
            
            # Use LLM to analyze topic diversity
            content_sample = " ".join([msg["content"][:100] for msg in messages[:20]])
            
            prompt = f"""Analyze the topic diversity in this Discord channel content on a scale of 0.0 to 1.0:

Content sample: "{content_sample}..."

Consider:
- Number of distinct topics discussed
- Variety of discussion themes
- Mix of content types (questions, announcements, discussions, etc.)

Return only a number between 0.0 and 1.0 where 1.0 means very diverse topics."""

            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=10,
                temperature=0.1
            )
            
            try:
                diversity = float(response.strip())
                return max(0.0, min(1.0, diversity))
            except ValueError:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error analyzing topic diversity: {e}")
            return 0.0
    
    async def _create_split_recommendation(self, channel_id: str, stats: Dict[str, Any], diversity: float) -> Optional[Recommendation]:
        """Create a split recommendation for a diverse channel."""
        try:
            prompt = f"""Create a channel split recommendation for a diverse Discord channel.

Channel: {channel_id}
Message Count: {stats['message_count']}
Unique Users: {stats['unique_users']}
Topic Diversity: {diversity:.2f}

Return a JSON object with:
- title: Short title for the recommendation
- description: Brief description of the split
- rationale: Why this channel should be split
- priority: "high", "medium", or "low"
- expected_impact: What impact this split would have"""

            response = await self.llm_client.generate_json(prompt=prompt)
            
            return Recommendation(
                id=f"split_{channel_id}",
                type="split",
                title=response.get("title", f"Split {channel_id}"),
                description=response.get("description", ""),
                channels_involved=[channel_id],
                rationale=response.get("rationale", ""),
                priority=response.get("priority", "medium"),
                expected_impact=response.get("expected_impact", "")
            )
            
        except Exception as e:
            logger.error(f"Error creating split recommendation: {e}")
            return None
    
    async def _find_reorganize_candidates(self, channel_stats: Dict[str, Any], similarities: Dict[str, Dict[str, float]]) -> List[Recommendation]:
        """Find general reorganization opportunities."""
        # This is a placeholder for more complex reorganization logic
        # Could include category suggestions, naming improvements, etc.
        return []
    
    async def process(self, state: AgentState) -> AgentState:
        """Process state through the structure agent."""
        recommendations = await self.run()
        
        # Update state with recommendations
        state["structure_result"] = recommendations
        state["response"] = self._format_structure_response(recommendations)
        
        return state
    
    def _format_structure_response(self, recommendations: List[Recommendation]) -> str:
        """Format structure recommendations as a readable response."""
        if not recommendations:
            return "No channel structure recommendations found."
        
        if "error" in recommendations[0]:
            return f"Error analyzing structure: {recommendations[0]['error']}"
        
        response = "## 🏗️ Channel Structure Recommendations\n\n"
        
        for rec in recommendations:
            priority_emoji = {
                "high": "🔴",
                "medium": "🟡", 
                "low": "🟢"
            }.get(rec["priority"], "🟡")
            
            type_emoji = {
                "merge": "🔗",
                "split": "✂️",
                "reorganize": "🔄"
            }.get(rec["type"], "💡")
            
            response += f"### {priority_emoji} {type_emoji} {rec['title']}\n"
            response += f"**Type:** {rec['type'].title()}\n"
            response += f"**Channels:** {', '.join(rec['channels_involved'])}\n"
            response += f"**Description:** {rec['description']}\n"
            response += f"**Rationale:** {rec['rationale']}\n"
            response += f"**Expected Impact:** {rec['expected_impact']}\n\n"
        
        return response
    
    def can_handle(self, task) -> bool:
        """Check if this agent can handle the given task."""
        return (task.task_type == "structure" or 
                "structure" in task.description.lower() or
                "channel" in task.description.lower() or
                "organize" in task.description.lower()) 