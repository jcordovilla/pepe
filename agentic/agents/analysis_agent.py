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
        
        # Analysis configuration - using local Llama model
        self.max_content_length = config.get("max_content_length", 8000)
        self.summary_length = config.get("summary_length", "medium")
        self.extract_insights = config.get("extract_insights", True)
        
        # Analysis patterns
        self.skill_patterns = self._compile_skill_patterns()
        self.topic_patterns = self._compile_topic_patterns()
        
        logger.info(f"AnalysisAgent initialized with local Llama model")
        
        # Register this agent
        agent_registry.register_agent(self)
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process the current state and return updated state.
        """
        try:
            subtask = state.get("current_subtask")
            if not subtask:
                self.logger.warning("No subtask provided to AnalysisAgent")
                return state
            task_type = subtask.task_type.lower()
            # Dummy handling for new types
            if task_type == "server_analysis":
                state["analysis_results"] = {"summary": "Server analysis completed (dummy result)", "details": {"active_channels": ["general", "random"], "peak_times": ["12:00", "18:00"]}}
                return state
            elif task_type == "user_analysis":
                state["analysis_results"] = {"summary": "User analysis completed (dummy result)", "details": {"top_users": ["alice", "bob"], "engagement": {"alice": 42, "bob": 37}}}
                return state
            elif task_type == "content_analysis":
                state["analysis_results"] = {"summary": "Content analysis completed (dummy result)", "details": {"topics": ["AI", "ethics"], "message_types": {"text": 120, "image": 5}}}
                return state
            # Fallback to existing logic
            return await super().process(state)
        except Exception as e:
            self.logger.error(f"Error in AnalysisAgent: {e}")
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"AnalysisAgent error: {str(e)}")
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
            "analyze", "summarize", "extract_insights", "classify_content", "extract_skills", "analyze_trends",
            "server_analysis", "user_analysis", "content_analysis"
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
        base_prompt = f"""You are a Discord conversation summarizer. Create a {summary_type} summary of this Discord chat:

{content}

REQUIREMENTS:
- Format for Discord: Use clear sections with **bold headers**
- Length: {self.summary_length} (concise but comprehensive)
- Include key discussion points and conclusions
- Mention important users when relevant: @username
- Highlight decisions, action items, and next steps
- Use bullet points for lists
- Keep tone professional but conversational

STRUCTURE:
**Key Topics Discussed**
- Main themes and subjects

**Important Points**
- Key insights and conclusions

**Decisions & Action Items**
- Any decisions made or tasks assigned

**Notable Participants**
- Key contributors and their roles

QUALITY CRITERIA:
- Accurate representation of the conversation
- Clear organization with logical flow
- Actionable insights when present
- Appropriate level of detail for the length"""
        
        if focus_areas:
            base_prompt += f"\n\nSPECIAL FOCUS: Emphasize {', '.join(focus_areas)} in your summary."
        
        return base_prompt
    
    async def _generate_llm_summary(self, prompt: str) -> str:
        """Generate summary using the unified LLM client."""
        try:
            from ..services.llm_client import get_llm_client
            llm_client = get_llm_client()
            
            # Generate summary using the same Llama model
            summary = await llm_client.generate(
                prompt=prompt,
                max_tokens=self.summary_length * 2,  # Allow for longer summaries
                temperature=0.3  # Slightly higher temperature for creative summaries
            )
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error generating LLM summary: {e}")
            return "Unable to generate summary due to LLM error."
    
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
        Generate a capability response for bot capability queries.
        
        Args:
            subtask: Capability response subtask
            state: Current agent state
            
        Returns:
            Comprehensive capability response
        """
        try:
            query = state.get("user_context", {}).get("query", "")
            query_lower = query.lower()
            
            # Check if this is a general capability query
            capability_keywords = ["capable", "capabilities", "features", "help", "what can", "how to use", "what does"]
            is_capability_query = any(keyword in query_lower for keyword in capability_keywords)
            
            if is_capability_query:
                return await self._generate_bot_capabilities_response()
            else:
                # For other queries, check if we have search results
                search_results = state.get("search_results", [])
                if search_results:
                    return await self._generate_results_response(query, search_results)
                else:
                    return await self._generate_no_results_response(query, state)
            
        except Exception as e:
            logger.error(f"Error generating capability response: {e}")
            return {
                "response": "I can help you search through Discord conversations. Try asking me to find specific topics, users, or messages.",
                "response_type": "fallback"
            }
    
    async def _generate_bot_capabilities_response(self) -> Dict[str, Any]:
        """Generate a comprehensive bot capabilities response."""
        try:
            response = """**🤖 Discord Bot Capabilities**

I'm an AI-powered Discord bot that can help you search, analyze, and understand your server conversations. Here's what I can do:

**🔍 Search & Discovery**
• **Semantic Search**: Find messages by meaning, not just keywords
• **Filtered Search**: Search by channel, user, time period, or reactions
• **Keyword Search**: Find specific terms or phrases
• **Resource Search**: Find shared links, files, and resources

**📊 Analysis & Insights**
• **Summarize**: Create summaries of discussions, channels, or time periods
• **Analyze Trends**: Identify patterns and trends in conversations
• **Extract Insights**: Find key themes and important points
• **Classify Content**: Categorize messages by topic, complexity, or urgency
• **Extract Skills**: Identify technologies and skills mentioned

**📈 Digests & Reports**
• **Weekly Digests**: Summarize a week's worth of activity
• **Monthly Digests**: Comprehensive monthly overviews
• **Custom Time Periods**: Analyze any specific time range

**💡 Smart Features**
• **Context Awareness**: Understand conversation context and history
• **Multi-Channel Analysis**: Compare discussions across channels
• **User Activity Analysis**: Track participation and engagement
• **Topic Tracking**: Follow discussions on specific subjects

**🎯 Example Queries**
• "Summarize last week's Python discussions"
• "Find messages about machine learning from @user123"
• "What are the trending topics this month?"
• "Show me resources shared in the help channel"
• "Analyze the sentiment of recent discussions"

**💬 How to Use**
Just ask me natural language questions! I'll interpret your intent and provide relevant information from your Discord conversations.

Need help with a specific query? Try asking me to search for topics, summarize discussions, or analyze patterns in your server!"""

            return {
                "response": response,
                "response_type": "capabilities",
                "capabilities": {
                    "search": ["semantic", "filtered", "keyword", "resource"],
                    "analysis": ["summarize", "trends", "insights", "classification"],
                    "digests": ["weekly", "monthly", "custom"],
                    "features": ["context_aware", "multi_channel", "user_analysis", "topic_tracking"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating bot capabilities response: {e}")
            return {
                "response": "I'm an AI Discord bot that can search, analyze, and summarize your server conversations. Try asking me to find specific topics or summarize recent discussions!",
                "response_type": "capabilities_fallback"
            }
    
    async def _generate_results_response(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a response based on search results."""
        try:
            # Extract key information from results
            result_count = len(results)
            top_result = results[0] if results else {}
            
            # Generate contextual response
            if result_count == 1:
                response = f"I found 1 message about '{query}':\n\n"
                response += f"**{top_result.get('author', {}).get('username', 'Unknown')}** in #{top_result.get('channel_name', 'unknown')}:\n"
                response += f"{top_result.get('content', '')[:200]}..."
            else:
                response = f"I found {result_count} messages about '{query}'. Here's the most relevant one:\n\n"
                response += f"**{top_result.get('author', {}).get('username', 'Unknown')}** in #{top_result.get('channel_name', 'unknown')}:\n"
                response += f"{top_result.get('content', '')[:200]}..."
                
                if result_count > 3:
                    response += f"\n\n... and {result_count - 1} more messages. Would you like me to show more results?"
            
            return {
                "response": response,
                "response_type": "results",
                "result_count": result_count,
                "top_result": top_result
            }
            
        except Exception as e:
            logger.error(f"Error generating results response: {e}")
            return {
                "response": f"I found {len(results)} messages about '{query}'. Here's what I found:",
                "response_type": "results_simple"
            }
    
    async def _generate_no_results_response(self, query: str, state: AgentState) -> Dict[str, Any]:
        """Generate a contextual response when no search results are found."""
        try:
            # Analyze the query to provide helpful suggestions
            query_lower = query.lower()
            
            # Extract potential topics from query
            topics = []
            if any(word in query_lower for word in ['python', 'code', 'programming']):
                topics.append('programming')
            if any(word in query_lower for word in ['ai', 'machine learning', 'ml']):
                topics.append('AI/ML')
            if any(word in query_lower for word in ['discord', 'bot', 'server']):
                topics.append('Discord')
            if any(word in query_lower for word in ['help', 'support', 'issue']):
                topics.append('support')
            
            # Generate contextual response
            if topics:
                response = f"I couldn't find any messages about '{query}' in our Discord conversations. "
                response += f"This might be because:\n"
                response += f"• The topic hasn't been discussed recently\n"
                response += f"• The search terms might be too specific\n"
                response += f"• The conversation might be in a different channel\n\n"
                response += f"Try searching for broader terms related to {', '.join(topics)} or ask about recent discussions."
            else:
                response = f"I couldn't find any messages about '{query}' in our Discord conversations. "
                response += f"Try:\n"
                response += f"• Using different keywords\n"
                response += f"• Searching for broader topics\n"
                response += f"• Asking about recent discussions in specific channels"
            
            # Add helpful examples based on query type
            if '?' in query:
                response += f"\n\nFor questions, try asking about recent discussions or popular topics in our channels."
            elif len(query.split()) <= 2:
                response += f"\n\nFor short queries, try adding more context or related terms."
            
            return {
                "response": response,
                "response_type": "no_results",
                "suggested_topics": topics,
                "query_analysis": {
                    "is_question": '?' in query,
                    "word_count": len(query.split()),
                    "has_specific_terms": len(topics) > 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating no results response: {e}")
            return {
                "response": f"I couldn't find any messages about '{query}'. Try using different keywords or asking about recent discussions.",
                "response_type": "no_results_fallback"
            }


class CapabilityAgent(BaseAgent):
    """
    Agent responsible for handling bot capability queries.
    Returns a static or mock description of the bot's features.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.CAPABILITY, config)
        agent_registry.register_agent(self)

    def can_handle(self, task: SubTask) -> bool:
        if not task or not task.task_type:
            return False
        return task.task_type.lower() == "capability_response"

    async def process(self, state: AgentState) -> AgentState:
        state["analysis_results"] = {
            "summary": "Bot Capabilities",
            "details": {
                "features": [
                    "Semantic search across Discord messages",
                    "Summarization of channel activity",
                    "User and channel analytics",
                    "Trending topics and digests",
                    "Custom time range queries",
                    "Performance and reliability reporting"
                ],
                "usage": "Ask questions about server activity, trends, or request summaries."
            }
        }
        return state

# Register the capability agent globally
CapabilityAgent({})
