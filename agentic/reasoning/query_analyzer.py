"""
Query Analyzer

Analyzes user queries to understand intent, extract entities, and determine
the appropriate execution strategy.
"""

import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from openai import OpenAI
import os

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
            ]
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            "channel": r"#[\w-]+|channel\s+[\w-]+|in\s+([\w-]+)",
            "user": r"@[\w.-]+|by\s+([\w.-]+)|from\s+([\w.-]+)",
            "time_range": r"(last|past|previous)\s+\w+|between\s+[\d-]+\s+and\s+[\d-]+",
            "keyword": r"'([^']+)'|\"([^\"]+)\"|about\s+(\w+)",
            "count": r"top\s+(\d+)|(\d+)\s+results|limit\s+(\d+)"
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
            
            # Enhance with LLM analysis for complex queries
            if analysis["complexity"] > 0.7:
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
                entity = {
                    "type": entity_type,
                    "value": match.group(1) if match.groups() else match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.8  # Rule-based confidence
                }
                entities.append(entity)
        
        return entities
    
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
            
            return {
                "enhanced_intent": enhanced.get("enhanced_intent"),
                "additional_entities": enhanced.get("additional_entities", []),
                "sub_queries": enhanced.get("sub_queries", []),
                "execution_hints": enhanced.get("execution_hints", [])
            }
            
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
