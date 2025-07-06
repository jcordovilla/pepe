"""
Analysis Agent

Specialized agent for content analysis, summarization, and insight extraction.
"""

from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
import asyncio
import re
from collections import Counter

from .base_agent import BaseAgent, AgentRole, AgentState, SubTask, TaskStatus, agent_registry

logger = logging.getLogger(__name__)


class AnalysisAgent(BaseAgent):
    """
    Agent responsible for content analysis and summarization.
    
    This agent:
    - Generates summaries from search results
    - Extracts insights and trends
    - Performs content classification
    - Identifies key topics and themes
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.ANALYZER, config)
        
        # Analysis configuration
        self.summarization_model = config.get("summarization_model", "gpt-4-turbo")
        self.max_content_length = config.get("max_content_length", 8000)
        self.summary_length = config.get("summary_length", "medium")
        self.extract_insights = config.get("extract_insights", True)
        
        # Analysis patterns
        self.skill_patterns = self._compile_skill_patterns()
        self.topic_patterns = self._compile_topic_patterns()
        
        logger.info(f"AnalysisAgent initialized with model={self.summarization_model}")
        
        # Register this agent
        agent_registry.register_agent(self)
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process analysis-related subtasks.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with analysis results
        """
        try:
            subtasks = state.get("subtasks", [])
            analysis_subtasks = [task for task in subtasks if self.can_handle(task)]
            
            if not analysis_subtasks:
                logger.warning("No analysis subtasks found")
                return state
            
            analysis_results = {}
            
            for subtask in analysis_subtasks:
                logger.info(f"Processing analysis subtask: {subtask.task_type}")
                
                if subtask.task_type == "summarize":
                    result = await self._summarize_content(subtask, state)
                elif subtask.task_type == "extract_insights":
                    result = await self._extract_insights(subtask, state)
                elif subtask.task_type == "classify_content":
                    result = await self._classify_content(subtask, state)
                elif subtask.task_type == "extract_skills":
                    result = await self._extract_skills(subtask, state)
                elif subtask.task_type == "analyze_trends":
                    result = await self._analyze_trends(subtask, state)
                elif subtask.task_type == "capability_response":
                    result = await self._generate_capability_response(subtask, state)
                else:
                    logger.warning(f"Unknown analysis task type: {subtask.task_type}")
                    continue
                
                analysis_results[subtask.task_type] = result
                
                # Update subtask status
                subtask.status = TaskStatus.COMPLETED
                subtask.result = result
            
            # Update state
            state["analysis_results"] = analysis_results
            state["metadata"]["analysis_agent"] = {
                "analysis_time": datetime.utcnow().isoformat(),
                "subtasks_processed": len(analysis_subtasks),
                "results_count": len(analysis_results)
            }
            
            logger.info(f"Analysis completed: {len(analysis_results)} results generated")
            return state
            
        except Exception as e:
            logger.error(f"Error in analysis agent: {e}")
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Analysis error: {str(e)}")
            return state
    
    def can_handle(self, task: SubTask) -> bool:
        """
        Determine if this agent can handle the given task.
        
        Args:
            task: Task to evaluate
            
        Returns:
            True if task is analysis-related
        """
        if not task or not task.task_type:
            return False
            
        analysis_types = [
            "summarize", "analyze", "extract", "classify",
            "insights", "trends", "skills", "topics", "capability_response"
        ]
        task_type = task.task_type.lower() if task.task_type else ""
        return any(analysis_type in task_type for analysis_type in analysis_types)
    
    async def _summarize_content(self, subtask: SubTask, state: AgentState) -> Dict[str, Any]:
        """
        Generate a summary from content or search results.
        
        Args:
            subtask: Summarization subtask
            state: Current agent state
            
        Returns:
            Summary results
        """
        try:
            # Get content to summarize
            content_source = subtask.parameters.get("content_source", "search_results")
            
            if content_source == "search_results":
                search_results = state.get("search_results", [])
                content = self._extract_content_from_results(search_results)
            else:
                content = subtask.parameters.get("content", "")
            
            if not content:
                return {"summary": "No content available for summarization.", "word_count": 0}
            
            # Prepare content for summarization
            prepared_content = self._prepare_content_for_summary(content)
            
            # Generate summary using LLM
            summary_prompt = self._build_summary_prompt(
                prepared_content,
                subtask.parameters.get("summary_type", "overview"),
                subtask.parameters.get("focus_areas", [])
            )
            
            # TODO: Replace with actual LLM call
            summary = await self._generate_llm_summary(summary_prompt)
            
            # Extract additional metadata
            metadata = {
                "content_length": len(content),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(content) if content else 0,
                "source_messages": len(state.get("search_results", [])),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return {
                "summary": summary,
                "metadata": metadata,
                "word_count": len(summary.split())
            }
            
        except Exception as e:
            logger.error(f"Error in content summarization: {e}")
            return {"summary": f"Error generating summary: {str(e)}", "word_count": 0}
    
    async def _extract_insights(self, subtask: SubTask, state: AgentState) -> Dict[str, Any]:
        """
        Extract insights and patterns from content.
        
        Args:
            subtask: Insight extraction subtask
            state: Current agent state
            
        Returns:
            Extracted insights
        """
        try:
            search_results = state.get("search_results", [])
            
            if not search_results:
                return {"insights": [], "patterns": [], "key_themes": []}
            
            # Extract various insights
            insights = {
                "key_themes": await self._extract_key_themes(search_results),
                "user_activity": await self._analyze_user_activity(search_results),
                "temporal_patterns": await self._analyze_temporal_patterns(search_results),
                "topic_distribution": await self._analyze_topic_distribution(search_results),
                "sentiment_analysis": await self._analyze_sentiment(search_results)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            return {"insights": [], "error": str(e)}
    
    async def _classify_content(self, subtask: SubTask, state: AgentState) -> Dict[str, Any]:
        """
        Classify content into categories.
        
        Args:
            subtask: Classification subtask
            state: Current agent state
            
        Returns:
            Classification results
        """
        try:
            search_results = state.get("search_results", [])
            
            classifications = {
                "by_type": self._classify_by_type(search_results),
                "by_topic": self._classify_by_topic(search_results),
                "by_complexity": self._classify_by_complexity(search_results),
                "by_urgency": self._classify_by_urgency(search_results)
            }
            
            return classifications
            
        except Exception as e:
            logger.error(f"Error in content classification: {e}")
            return {"classifications": {}, "error": str(e)}
    
    async def _extract_skills(self, subtask: SubTask, state: AgentState) -> Dict[str, Any]:
        """
        Extract skills and technologies mentioned in content.
        
        Args:
            subtask: Skill extraction subtask
            state: Current agent state
            
        Returns:
            Extracted skills
        """
        try:
            search_results = state.get("search_results", [])
            
            all_skills = []
            skill_contexts = {}
            
            for result in search_results:
                content = result.get("content", "")
                author = result.get("author", {}).get("username", "unknown")
                
                # Extract skills using patterns
                found_skills = self._extract_skills_from_text(content)
                
                for skill in found_skills:
                    all_skills.append(skill)
                    if skill not in skill_contexts:
                        skill_contexts[skill] = []
                    
                    skill_contexts[skill].append({
                        "author": author,
                        "context": content[:200],  # First 200 chars for context
                        "timestamp": result.get("timestamp", "")
                    })
            
            # Count and rank skills
            skill_counts = Counter(all_skills)
            
            return {
                "skills": dict(skill_counts.most_common(20)),
                "skill_contexts": skill_contexts,
                "total_skills_found": len(all_skills),
                "unique_skills": len(skill_counts)
            }
            
        except Exception as e:
            logger.error(f"Error extracting skills: {e}")
            return {"skills": {}, "error": str(e)}
    
    async def _analyze_trends(self, subtask: SubTask, state: AgentState) -> Dict[str, Any]:
        """
        Analyze trends in the content.
        
        Args:
            subtask: Trend analysis subtask
            state: Current agent state
            
        Returns:
            Trend analysis results
        """
        try:
            search_results = state.get("search_results", [])
            
            trends = {
                "temporal_trends": await self._analyze_temporal_trends(search_results),
                "topic_trends": await self._analyze_topic_trends(search_results),
                "user_engagement_trends": await self._analyze_engagement_trends(search_results),
                "content_volume_trends": await self._analyze_volume_trends(search_results)
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {"trends": {}, "error": str(e)}
    
    def _extract_content_from_results(self, results: List[Dict[str, Any]]) -> str:
        """Extract and combine content from search results."""
        content_parts = []
        
        for result in results:
            author = result.get("author", {}).get("username", "Unknown")
            timestamp = result.get("timestamp", "")
            content = result.get("content", "")
            
            if content.strip():
                content_parts.append(f"[{author} - {timestamp}]: {content}")
        
        return "\n\n".join(content_parts)
    
    def _prepare_content_for_summary(self, content: str) -> str:
        """Prepare content for summarization by cleaning and truncating."""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Truncate if too long
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "..."
        
        return content.strip()
    
    def _build_summary_prompt(self, content: str, summary_type: str, focus_areas: List[str]) -> str:
        """Build prompt for LLM summarization."""
        base_prompt = f"""
Please provide a {summary_type} summary of the following Discord conversation content:

{content}

Summary requirements:
- Length: {self.summary_length}
- Focus on key discussion points and conclusions
- Maintain context about who said what when relevant
- Highlight important decisions or action items
"""
        
        if focus_areas:
            base_prompt += f"\nSpecial focus on: {', '.join(focus_areas)}"
        
        return base_prompt
    
    async def _generate_llm_summary(self, prompt: str) -> str:
        """Generate summary using LLM (placeholder for actual implementation)."""
        # TODO: Implement actual LLM call
        return "Summary placeholder - implement LLM integration"
    
    def _compile_skill_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for skill extraction."""
        skills = [
            r'\b(python|javascript|typescript|java|c\+\+|rust|go|kotlin)\b',
            r'\b(react|vue|angular|svelte|nextjs|nuxt)\b',
            r'\b(docker|kubernetes|aws|azure|gcp|terraform)\b',
            r'\b(postgres|mysql|mongodb|redis|elasticsearch)\b',
            r'\b(git|github|gitlab|bitbucket|cicd|jenkins)\b'
        ]
        
        return [re.compile(pattern, re.IGNORECASE) for pattern in skills]
    
    def _compile_topic_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for topic classification."""
        topics = {
            "technical": re.compile(r'\b(code|programming|development|bug|error|api|database)\b', re.IGNORECASE),
            "discussion": re.compile(r'\b(think|opinion|believe|discuss|consider|thoughts)\b', re.IGNORECASE),
            "question": re.compile(r'\b(how|what|why|when|where|can|could|should|would)\b', re.IGNORECASE),
            "resource": re.compile(r'\b(link|url|documentation|tutorial|guide|article)\b', re.IGNORECASE)
        }
        
        return topics
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text using patterns."""
        skills = []
        
        for pattern in self.skill_patterns:
            matches = pattern.findall(text)
            skills.extend(matches)
        
        return skills
    
    def _classify_by_type(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Classify messages by type."""
        type_counts = Counter()
        
        for result in results:
            content = result.get("content", "").lower()
            
            for topic_type, pattern in self.topic_patterns.items():
                if pattern.search(content):
                    type_counts[topic_type] += 1
        
        return dict(type_counts)
    
    def _classify_by_topic(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Classify messages by topic."""
        # Simple keyword-based topic classification
        topics = {
            "ai_ml": ["ai", "machine learning", "neural", "model", "training"],
            "web_dev": ["frontend", "backend", "api", "web", "server"],
            "devops": ["docker", "kubernetes", "deployment", "ci/cd"],
            "data": ["database", "sql", "analytics", "visualization"]
        }
        
        topic_counts = Counter()
        
        for result in results:
            content = result.get("content", "").lower()
            
            for topic, keywords in topics.items():
                if any(keyword in content for keyword in keywords):
                    topic_counts[topic] += 1
        
        return dict(topic_counts)
    
    def _classify_by_complexity(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Classify messages by complexity level."""
        complexity_counts = Counter()
        
        for result in results:
            content = result.get("content", "")
            
            # Simple complexity heuristics
            if len(content) > 500:
                complexity_counts["high"] += 1
            elif len(content) > 100:
                complexity_counts["medium"] += 1
            else:
                complexity_counts["low"] += 1
        
        return dict(complexity_counts)
    
    def _classify_by_urgency(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Classify messages by urgency level."""
        urgency_patterns = {
            "urgent": re.compile(r'\b(urgent|asap|immediately|critical|emergency)\b', re.IGNORECASE),
            "medium": re.compile(r'\b(soon|quickly|priority|important)\b', re.IGNORECASE)
        }
        
        urgency_counts = Counter()
        
        for result in results:
            content = result.get("content", "")
            
            if urgency_patterns["urgent"].search(content):
                urgency_counts["urgent"] += 1
            elif urgency_patterns["medium"].search(content):
                urgency_counts["medium"] += 1
            else:
                urgency_counts["normal"] += 1
        
        return dict(urgency_counts)
    
    async def _extract_key_themes(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract key themes from results."""
        # Simple keyword frequency analysis
        all_words = []
        
        for result in results:
            content = result.get("content", "")
            words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        
        # Filter out common words and return top themes
        common_words = {"that", "this", "with", "have", "will", "from", "they", "been", "said", "each", "which", "their", "time", "about"}
        themes = [word for word, count in word_counts.most_common(10) if word not in common_words]
        
        return themes[:5]
    
    async def _analyze_user_activity(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user activity patterns."""
        user_counts = Counter()
        
        for result in results:
            author = result.get("author", {}).get("username", "unknown")
            user_counts[author] += 1
        
        return {
            "most_active_users": dict(user_counts.most_common(5)),
            "total_unique_users": len(user_counts),
            "average_messages_per_user": sum(user_counts.values()) / len(user_counts) if user_counts else 0
        }
    
    async def _analyze_temporal_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in messages."""
        hour_counts = Counter()
        day_counts = Counter()
        
        for result in results:
            timestamp = result.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    hour_counts[dt.hour] += 1
                    day_counts[dt.strftime("%A")] += 1
                except:
                    pass
        
        return {
            "peak_hours": dict(hour_counts.most_common(3)),
            "peak_days": dict(day_counts.most_common(3)),
            "total_time_span": len(hour_counts)
        }
    
    async def _analyze_topic_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze distribution of topics."""
        return self._classify_by_topic(results)
    
    async def _analyze_sentiment(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment of messages (simplified)."""
        positive_words = ["good", "great", "awesome", "excellent", "amazing", "love", "like", "happy"]
        negative_words = ["bad", "terrible", "awful", "hate", "dislike", "sad", "angry", "frustrated"]
        
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        
        for result in results:
            content = result.get("content", "").lower()
            
            positive_score = sum(1 for word in positive_words if word in content)
            negative_score = sum(1 for word in negative_words if word in content)
            
            if positive_score > negative_score:
                sentiment_counts["positive"] += 1
            elif negative_score > positive_score:
                sentiment_counts["negative"] += 1
            else:
                sentiment_counts["neutral"] += 1
        
        return sentiment_counts
    
    async def _analyze_temporal_trends(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends over time."""
        return await self._analyze_temporal_patterns(results)
    
    async def _analyze_topic_trends(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how topics trend over time."""
        return {"topic_trends": "Not implemented yet"}
    
    async def _analyze_engagement_trends(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user engagement trends."""
        return await self._analyze_user_activity(results)
    
    async def _analyze_volume_trends(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze content volume trends."""
        return {"volume_trends": f"Total messages analyzed: {len(results)}"}
    
    async def _generate_capability_response(self, subtask: SubTask, state: AgentState) -> Dict[str, Any]:
        """
        Generate a comprehensive capability response for meta-queries.
        
        Args:
            subtask: Capability response subtask
            state: Current agent state
            
        Returns:
            Capability information and help documentation
        """
        try:
            query = subtask.parameters.get("query", "")
            
            # Build comprehensive capability response
            capabilities = {
                "search_capabilities": [
                    "**Message Search**: Find specific messages by content, author, or channel",
                    "**Semantic Search**: Understand context and find related discussions",
                    "**Filtered Search**: Search within specific channels, time ranges, or by specific users",
                    "**Keyword Search**: Find messages containing specific terms or phrases"
                ],
                "analysis_capabilities": [
                    "**Content Summarization**: Create summaries of conversations or topics",
                    "**Trend Analysis**: Identify patterns in discussions and user activity",
                    "**Topic Classification**: Categorize messages by type and subject",
                    "**Insight Extraction**: Find key themes and important information",
                    "**User Activity Analysis**: Track participation and engagement patterns"
                ],
                "special_features": [
                    "**Reaction Analysis**: Find most-reacted or popular messages",
                    "**Resource Detection**: Identify shared links, documents, and tools",
                    "**Time-based Queries**: Search within specific time periods",
                    "**Channel-specific Search**: Focus on particular channels or topics",
                    "**Cross-conversation Context**: Understand discussions across multiple messages"
                ],
                "data_sources": [
                    "**Discord Messages**: Access to historical Discord conversations",
                    "**User Interactions**: Track of user activities and patterns",
                    "**Shared Resources**: Links, documents, and tools shared in conversations",
                    "**Reaction Data**: Community engagement through reactions and responses"
                ]
            }
            
            # Generate contextual response based on query
            response_parts = []
            
            # Main capabilities introduction
            response_parts.append("## 🤖 AI Assistant Capabilities")
            response_parts.append("I'm an intelligent Discord bot that can help you explore and analyze your server's conversations. Here's what I can do:")
            
            # Add search capabilities
            response_parts.append("\n### 🔍 **Search & Discovery**")
            response_parts.extend(capabilities["search_capabilities"])
            
            # Add analysis capabilities
            response_parts.append("\n### 📊 **Analysis & Insights**")
            response_parts.extend(capabilities["analysis_capabilities"])
            
            # Add special features
            response_parts.append("\n### ⚡ **Special Features**")
            response_parts.extend(capabilities["special_features"])
            
            # Add data sources
            response_parts.append("\n### 📁 **Data Sources**")
            response_parts.extend(capabilities["data_sources"])
            
            # Add examples section
            response_parts.append("\n### 💡 **Example Questions You Can Ask**")
            examples = [
                "• `Find messages about AI from last week`",
                "• `Summarize the discussion in #general channel`",
                "• `What are the most popular topics this month?`",
                "• `Show me messages with the most reactions`",
                "• `Find resources shared about Python programming`",
                "• `Who are the most active users in #development?`",
                "• `What was discussed about machine learning recently?`",
                "• `Give me a digest of last week's conversations`"
            ]
            response_parts.extend(examples)
            
            # Add usage tips
            response_parts.append("\n### 📝 **Usage Tips**")
            tips = [
                "• **Be specific**: Include channel names, timeframes, or topics for better results",
                "• **Use natural language**: Ask questions as you would to a colleague",
                "• **Combine filters**: Search by channel, user, and time period together",
                "• **Ask follow-ups**: Build on previous queries for deeper insights"
            ]
            response_parts.extend(tips)
            
            # Add technical details
            response_parts.append("\n### 🛠 **Technical Details**")
            tech_details = [
                "• **AI-Powered**: Uses advanced language models for understanding context",
                "• **Vector Search**: Semantic similarity matching for relevant results",
                "• **Multi-Agent System**: Specialized agents for different types of analysis",
                "• **Real-time Processing**: Continuously indexes new messages as they arrive"
            ]
            response_parts.extend(tech_details)
            
            # Final response
            response_parts.append("\n---")
            response_parts.append("💬 **Ready to help!** Ask me anything about your Discord server's conversations and I'll provide intelligent, contextual responses.")
            
            full_response = "\n".join(response_parts)
            
            return {
                "capability_response": full_response,
                "response_type": "capability",
                "generated_at": datetime.utcnow().isoformat(),
                "sections": {
                    "search": capabilities["search_capabilities"],
                    "analysis": capabilities["analysis_capabilities"],
                    "features": capabilities["special_features"],
                    "data_sources": capabilities["data_sources"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating capability response: {e}")
            return {
                "capability_response": f"I'm an AI assistant that can help you search and analyze Discord conversations. I can find messages, create summaries, analyze trends, and provide insights about your server's discussions. However, I encountered an error generating the full capability information: {str(e)}",
                "response_type": "capability",
                "error": str(e)
            }
