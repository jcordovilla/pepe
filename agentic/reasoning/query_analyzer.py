"""
Query Analyzer

Analyzes user queries to understand intent, extract entities, and determine
the appropriate execution strategy.
"""

import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import hashlib

from openai import OpenAI
import os

from ..cache.smart_cache import SmartCache

from ..services.channel_resolver import ChannelResolver

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Analyzes user queries to understand intent and extract structured information.
    
    Uses both rule-based patterns and LLM-based analysis for comprehensive
    query understanding.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = config.get("model", "gpt-4-turbo")

        # Initialize cache for LLM enhancements
        self.cache = SmartCache(config.get("cache", {}))
        self.analysis_cache_ttl = int(config.get("analysis_cache_ttl", 86400))
        self.llm_complexity_threshold = float(
            config.get("llm_complexity_threshold", 0.85)
        )
        
        # Initialize channel resolver
        chromadb_path = config.get("chromadb_path", "./data/chromadb/chroma.sqlite3")
        self.channel_resolver = ChannelResolver(chromadb_path)
        
        # Intent patterns
        self.intent_patterns = {
            "search": [
                r"find|search|look for|show me|get|retrieve",
                r"messages? (about|containing|with|mentioning)",
                r"what.*said|who.*mentioned"
            ],
            "summarize": [
                r"summarize|summary|sum up|overview",
                r"what (happened|was discussed|were the topics)",
                r"key points|main topics|highlights"
            ],
            "analyze": [
                r"analyze|analysis|insights|patterns",
                r"trends|statistics|metrics",
                r"compare|contrast|differences"
            ],
            "resource_search": [
                r"links|resources|files|documents",
                r"papers|articles|tutorials|tools",
                r"shared.*resources|posted.*links"
            ],
            "data_availability": [
                r"what data|how much data|data available",
                r"how many messages|message count",
                r"date range|time range|coverage"
            ],
            "reactions": [
                r"most reacted|popular messages|top reaction",
                r"reactions to|emoji reactions|most liked",
                r"messages with (most|highest|top) reactions"
            ]
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            "channel": r"<#(\d+)>|(?:in|from)\s+(?:the\s+)?#?([\w-]+(?:\s+[\w-]+)*)\s+channel|#([\w🦾🤖🏛🗂❌💻📚🛠❓🌎🏘👋💠-]+)|(?:in|from)\s+([\w-]+-(?:ops|dev|agents|chat|help|support|resources))",
            "user": r"@[\w.-]+|by\s+([\w.-]+)|from\s+([\w.-]+)",
            "time_range": r"(last|past|previous)\s+\w+|between\s+[\d-]+\s+and\s+[\d-]+",
            "keyword": r"'([^']+)'|\"([^\"]+)\"|about\s+(\w+)",
            "count": r"(\d+)\s+(last|recent|messages)|top\s+(\d+)|(\d+)\s+results|limit\s+(\d+)",
            "reaction": r"(👍|❤️|😂|👀|🎉|🚀|👏|👌)|(:\w+:)|emoji|reaction"
        }
        
        # Combined patterns for enhanced analysis
        self.combined_patterns = {
            "intent": r"(?:search|find|get|fetch|show|list|give|tell|what|how|when|where|who|digest|summary|summarize|analyze)",
            "channel": r"<#(\d+)>|(?:in|from)\s+(?:the\s+)?#?([\w-]+(?:\s+[\w-]+)*)\s+channel|#([\w🦾🤖🏛🗂❌💻📚🛠❓🌎🏘👋💠-]+)|(?:in|from)\s+([\w-]+-(?:ops|dev|agents|chat|help|support|resources))",
            "user": r"(?:by|from)\s+(?:user\s+)?@?(\w+)|<@!?(\d+)>",
            "count": r"(?:last|first|recent|latest|newest)\s+(\d+)|(\d+)\s+(?:messages?|results?)",
            "temporal": r"(weekly|daily|monthly|yesterday|today|this\s+week|last\s+week|past\s+week|this\s+month|last\s+month|past\s+month)\s*(?:digest|summary|report)?",
            "digest": r"(digest|summary|report|overview|recap)\s*(?:of|for)?\s*(?:the|this|last)?\s*(week|month|day|period)?",
            "timeframe": r"(?:in|from|during)\s+(?:the\s+)?(?:last|past|recent)\s+(\d+)\s+(days?|weeks?|months?|hours?)",
            "content_type": r"(discussions?|messages?|posts?|content|activity|highlights?|key\s+points?|trending|popular)"
        }
        
        logger.info("Query analyzer initialized")
    
    async def analyze(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze a user query to extract intent, entities, and execution hints.
        
        Args:
            query: User's natural language query
            context: Additional context (user, channel, etc.)
            
        Returns:
            Analysis results with intent, entities, and metadata
        """
        try:
            analysis = {
                "query": query,
                "intent": await self._detect_intent(query),
                "entities": await self._extract_entities(query),
                "complexity": self._assess_complexity(query),
                "context": context or {},
                "timestamp": datetime.utcnow().isoformat()
            }

            analysis["grouped_entities"] = self._group_entities(analysis["entities"])
            
            # Enhance with LLM analysis for complex queries
            if analysis["complexity"] > self.llm_complexity_threshold:
                llm_analysis = await self._llm_enhance_analysis(query, analysis)
                analysis.update(llm_analysis)
            
            logger.info(f"Query analyzed: intent={analysis['intent']}, entities={len(analysis['entities'])}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            return {
                "query": query,
                "intent": "unknown",
                "entities": [],
                "complexity": 0.0,
                "error": str(e)
            }
    
    async def _detect_intent(self, query: str) -> str:
        """Detect the primary intent of the query"""
        query_lower = query.lower()
        
        # Check patterns for each intent
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        # Default intent based on query structure
        if "?" in query:
            return "search"
        elif any(word in query_lower for word in ["summarize", "summary", "what happened"]):
            return "summarize"
        else:
            return "search"  # Default to search
    
    async def _extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract structured entities from the query"""
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                # For patterns with multiple groups, get the first non-None group
                if match.groups():
                    value = next((g for g in match.groups() if g), match.group(0))
                else:
                    value = match.group(0)
                
                # Clean up the value
                if entity_type == "channel":
                    # Get all groups to understand what we matched
                    groups = match.groups()
                    
                    # Check if the first group (Discord mention ID) is populated
                    if groups[0]:  # This is a Discord mention <#ID>
                        channel_id = groups[0]  # The captured ID
                        
                        # Use our new method to resolve ID to name
                        from ..services.channel_resolver import ChannelResolver
                        resolver = ChannelResolver()
                        resolved_name = resolver.resolve_channel_id_to_name(channel_id)
                        
                        if resolved_name:
                            entity = {
                                "type": "channel",
                                "value": resolved_name,  # Use resolved name
                                "channel_id": channel_id,  # Keep the ID
                                "start": match.start(),
                                "end": match.end(),
                                "confidence": 0.95  # Very high confidence for Discord mentions
                            }
                        else:
                            # Channel ID not found in our data
                            entity = {
                                "type": "channel",
                                "value": f"channel-{channel_id}",  # Fallback name
                                "channel_id": channel_id,
                                "start": match.start(),
                                "end": match.end(),
                                "confidence": 0.5  # Medium confidence - valid format but not in our data
                            }
                    else:
                        # Regular channel name (not a Discord mention)
                        # value is from one of the other groups
                        channel_name = value.lstrip("#").strip()
                        
                        # Resolve channel name to ID using existing method
                        channel_id = self.channel_resolver.resolve_channel_name(channel_name)
                        
                        if channel_id:
                            # Store both channel_id and original name
                            entity = {
                                "type": "channel",
                                "value": channel_name,  # Keep original for display
                                "channel_id": channel_id,  # Add resolved ID
                                "start": match.start(),
                                "end": match.end(),
                                "confidence": 0.9  # High confidence for resolved channels
                            }
                        else:
                            # Channel not found, keep as is but lower confidence
                            entity = {
                                "type": "channel",
                                "value": channel_name,
                                "channel_id": None,
                                "start": match.start(),
                                "end": match.end(),
                                "confidence": 0.3  # Low confidence for unresolved channels
                            }
                elif entity_type == "user":
                    value = value.lstrip("@").strip()
                    entity = {
                        "type": entity_type,
                        "value": value,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.8  # Rule-based confidence
                    }
                elif entity_type == "count":
                    # Extract just the number
                    count_match = re.search(r'\d+', value)
                    if count_match:
                        value = count_match.group(0)
                    entity = {
                        "type": entity_type,
                        "value": value,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.8  # Rule-based confidence
                    }
                elif entity_type == "time_range":
                    parsed = self._parse_time_range(value)
                    if parsed:
                        entity = {
                            "type": entity_type,
                            "value": value,
                            "start": match.start(),
                            "end": match.end(),
                            "confidence": 0.9,
                            **parsed,
                        }
                    else:
                        entity = {
                            "type": entity_type,
                            "value": value,
                            "start": match.start(),
                            "end": match.end(),
                            "confidence": 0.5,
                        }
                else:
                    entity = {
                        "type": entity_type,
                        "value": value,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.8  # Rule-based confidence
                    }
                
                entities.append(entity)
        
        return entities

    def _parse_time_range(self, text: str) -> Optional[Dict[str, str]]:
        """Parse a natural language time expression into a start/end range."""
        try:
            value = text.lower().strip()
            now = datetime.utcnow()

            # Simple relative ranges
            if value in {"yesterday"}:
                start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                end = start + timedelta(days=1)
            elif value in {"today"}:
                start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                end = now
            elif value in {"last week", "past week", "previous week"}:
                start = now - timedelta(days=7)
                end = now
            elif value in {"last month", "past month", "previous month"}:
                start = now - timedelta(days=30)
                end = now
            else:
                m = re.search(r"(?:last|past)\s+(\d+)\s+days?", value)
                if m:
                    start = now - timedelta(days=int(m.group(1)))
                    end = now
                else:
                    m = re.search(r"(?:last|past)\s+(\d+)\s+weeks?", value)
                    if m:
                        start = now - timedelta(weeks=int(m.group(1)))
                        end = now
                    else:
                        m = re.search(r"(?:last|past)\s+(\d+)\s+months?", value)
                        if m:
                            start = now - timedelta(days=30 * int(m.group(1)))
                            end = now
                        else:
                            m = re.search(r"between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})", value)
                            if m:
                                start = datetime.fromisoformat(m.group(1))
                                end = datetime.fromisoformat(m.group(2))
                            else:
                                return None

            return {"start": start.isoformat(), "end": end.isoformat()}
        except Exception:
            return None

    def _group_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Group extracted entities by type for easy access."""
        grouped: Dict[str, Any] = {}
        for ent in entities:
            etype = ent["type"]
            if etype == "time_range":
                grouped["time_range"] = {"start": ent.get("start"), "end": ent.get("end")}
            else:
                key = f"{etype}s"
                grouped.setdefault(key, []).append(ent.get("value"))
        return grouped
    
    def _assess_complexity(self, query: str) -> float:
        """Assess query complexity (0.0 to 1.0)"""
        complexity_factors = {
            "length": len(query.split()) / 50.0,  # Normalize by 50 words
            "operators": len(re.findall(r'\b(and|or|not|but|however)\b', query.lower())) * 0.2,
            "time_references": len(re.findall(r'\b(yesterday|today|last week|between)\b', query.lower())) * 0.1,
            "multiple_intents": self._count_intents(query) * 0.3
        }
        
        total_complexity = sum(complexity_factors.values())
        return min(total_complexity, 1.0)
    
    def _count_intents(self, query: str) -> int:
        """Count how many different intents might be present"""
        query_lower = query.lower()
        intent_count = 0
        
        for intent, patterns in self.intent_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                intent_count += 1
        
        return intent_count
    
    async def _llm_enhance_analysis(self, query: str, base_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to enhance analysis for complex queries"""
        try:
            cache_key = f"llm_analysis:{hashlib.sha256(query.encode()).hexdigest()}"
            cached = await self.cache.get(cache_key)
            if cached:
                return cached

            system_prompt = """
            You are a query analysis expert. Analyze the user query and provide enhanced insights.
            Focus on:
            1. Refined intent classification
            2. Additional entity extraction
            3. Query decomposition suggestions
            4. Execution hints
            
            Return JSON with: enhanced_intent, additional_entities, sub_queries, execution_hints
            """
            
            user_prompt = f"""
            Query: "{query}"
            Base Analysis: {base_analysis}
            
            Provide enhanced analysis as JSON.
            """
            
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            import json
            enhanced = json.loads(response.choices[0].message.content)

            result = {
                "enhanced_intent": enhanced.get("enhanced_intent"),
                "additional_entities": enhanced.get("additional_entities", []),
                "sub_queries": enhanced.get("sub_queries", []),
                "execution_hints": enhanced.get("execution_hints", [])
            }

            await self.cache.set(cache_key, result, ttl=self.analysis_cache_ttl)
            return result

        except Exception as e:
            logger.error(f"Error in LLM enhancement: {str(e)}")
            return {}
    
    def get_entity_by_type(self, entities: List[Dict[str, Any]], entity_type: str) -> Optional[Dict[str, Any]]:
        """Get the first entity of a specific type"""
        for entity in entities:
            if entity["type"] == entity_type:
                return entity
        return None
    
    def get_entities_by_type(self, entities: List[Dict[str, Any]], entity_type: str) -> List[Dict[str, Any]]:
        """Get all entities of a specific type"""
        return [entity for entity in entities if entity["type"] == entity_type]
