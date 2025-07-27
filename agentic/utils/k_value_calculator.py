"""
Dynamic K-Value Calculator

Analyzes query nature and breadth to determine appropriate k values for vector retrieval.
Supports queries requiring 100+ results for comprehensive analysis.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class KValueCalculator:
    """
    Calculates appropriate k values based on query analysis.
    
    Features:
    - Query type classification
    - Time range analysis
    - Scope detection (single channel vs cross-server)
    - Complexity assessment
    - Performance-aware limits
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("k_value_config", {})
        self.base_values = self.config.get("base_values", {})
        self.multipliers = self.config.get("multipliers", {})
        self.query_patterns = self.config.get("query_patterns", {})
        self.max_values = self.config.get("max_values", {})
        self.performance = self.config.get("performance", {})
        
        logger.info("KValueCalculator initialized with dynamic configuration")
    
    def calculate_k_value(
        self, 
        query: str, 
        query_type: str = "search",
        entities: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate appropriate k value based on query analysis.
        
        Args:
            query: User query text
            query_type: Type of query (search, analysis, digest, etc.)
            entities: Extracted entities from query interpretation
            context: Additional context information
            
        Returns:
            Dict containing calculated k value and analysis details
        """
        try:
            # Start with base k value for query type
            base_k = self._get_base_k_value(query_type)
            
            # Analyze query characteristics
            query_analysis = self._analyze_query(query, entities, context)
            
            # Calculate multipliers
            time_multiplier = self._calculate_time_multiplier(query_analysis)
            scope_multiplier = self._calculate_scope_multiplier(query_analysis)
            complexity_multiplier = self._calculate_complexity_multiplier(query_analysis)
            
            # Apply multipliers
            calculated_k = int(base_k * time_multiplier * scope_multiplier * complexity_multiplier)
            
            # Apply maximum limits
            max_k = self._get_max_k_value(query_type, query_analysis)
            final_k = min(calculated_k, max_k)
            
            # Ensure minimum k value
            final_k = max(final_k, 5)
            
            result = {
                "k_value": final_k,
                "base_k": base_k,
                "multipliers": {
                    "time": time_multiplier,
                    "scope": scope_multiplier,
                    "complexity": complexity_multiplier,
                    "total": time_multiplier * scope_multiplier * complexity_multiplier
                },
                "analysis": query_analysis,
                "limits": {
                    "calculated": calculated_k,
                    "max_allowed": max_k,
                    "final": final_k
                }
            }
            
            logger.info(f"K-value calculation: {final_k} (base: {base_k}, multipliers: {result['multipliers']['total']:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating k value: {e}")
            # Fallback to safe default
            return {
                "k_value": 10,
                "base_k": 10,
                "multipliers": {"time": 1.0, "scope": 1.0, "complexity": 1.0, "total": 1.0},
                "analysis": {"query_type": "fallback"},
                "limits": {"calculated": 10, "max_allowed": 100, "final": 10}
            }
    
    def _get_base_k_value(self, query_type: str) -> int:
        """Get base k value for query type."""
        base_values = {
            "search": self.base_values.get("simple_search", 10),
            "semantic_search": self.base_values.get("simple_search", 10),
            "keyword_search": self.base_values.get("simple_search", 10),
            "filtered_search": self.base_values.get("detailed_search", 25),
            "analysis": self.base_values.get("analysis", 75),
            "skill_experience": self.base_values.get("skill_experience", 100),
            "trend_analysis": self.base_values.get("trend_analysis", 100),
            "digest": self.base_values.get("digest", 150),
            "weekly_digest": self.base_values.get("digest", 150),
            "monthly_digest": self.base_values.get("digest", 150),
            "user_analysis": self.base_values.get("analysis", 75),
            "content_analysis": self.base_values.get("analysis", 75),
            "server_analysis": self.base_values.get("cross_server_analysis", 200),
            "comprehensive_search": self.base_values.get("comprehensive_search", 50)
        }
        
        return base_values.get(query_type, self.base_values.get("simple_search", 10))
    
    def _analyze_query(
        self, 
        query: str, 
        entities: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze query characteristics for k-value calculation."""
        query_lower = query.lower()
        
        analysis = {
            "query_type": "search",
            "time_range": "recent",
            "scope": "single_channel",
            "complexity": "simple",
            "patterns_detected": [],
            "entities_found": []
        }
        
        # Detect query patterns
        for pattern_type, keywords in self.query_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                analysis["patterns_detected"].append(pattern_type)
                if pattern_type in ["comprehensive_analysis", "broad_search"]:
                    analysis["query_type"] = "analysis"
                    analysis["complexity"] = "complex"
                elif pattern_type == "skill_experience":
                    analysis["query_type"] = "skill_experience"
                    analysis["complexity"] = "complex"
        
        # Analyze time references
        time_patterns = {
            "today": ["today", "this morning", "tonight"],
            "yesterday": ["yesterday", "last night"],
            "this_week": ["this week", "current week", "week so far"],
            "last_week": ["last week", "previous week", "past week"],
            "this_month": ["this month", "current month", "month so far"],
            "last_month": ["last month", "previous month", "past month"],
            "all_time": ["all time", "ever", "always", "history", "everything"]
        }
        
        for time_key, patterns in time_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                analysis["time_range"] = time_key
                break
        
        # Analyze scope from entities
        if entities:
            channel_count = sum(1 for e in entities if e.get("type") == "channel")
            user_count = sum(1 for e in entities if e.get("type") == "user")
            
            if channel_count > 1:
                analysis["scope"] = "multiple_channels"
            elif channel_count == 1:
                analysis["scope"] = "single_channel"
            
            if "server" in query_lower or "all channels" in query_lower:
                analysis["scope"] = "all_channels"
            
            analysis["entities_found"] = [e.get("type") for e in entities]
        
        # Assess complexity
        complexity_indicators = {
            "simple": ["find", "search", "look for", "get"],
            "moderate": ["analyze", "show", "list", "what"],
            "complex": ["trends", "patterns", "insights", "analysis", "overview"],
            "very_complex": ["comprehensive", "detailed", "thorough", "complete analysis"]
        }
        
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                analysis["complexity"] = complexity
                break
        
        return analysis
    
    def _calculate_time_multiplier(self, analysis: Dict[str, Any]) -> float:
        """Calculate time-based multiplier."""
        time_range = analysis.get("time_range", "recent")
        time_multipliers = self.multipliers.get("time_range", {})
        return time_multipliers.get(time_range, 1.0)
    
    def _calculate_scope_multiplier(self, analysis: Dict[str, Any]) -> float:
        """Calculate scope-based multiplier."""
        scope = analysis.get("scope", "single_channel")
        scope_multipliers = self.multipliers.get("scope", {})
        return scope_multipliers.get(scope, 1.0)
    
    def _calculate_complexity_multiplier(self, analysis: Dict[str, Any]) -> float:
        """Calculate complexity-based multiplier."""
        complexity = analysis.get("complexity", "simple")
        complexity_multipliers = self.multipliers.get("complexity", {})
        return complexity_multipliers.get(complexity, 1.0)
    
    def _get_max_k_value(self, query_type: str, analysis: Dict[str, Any]) -> int:
        """Get maximum k value based on query type and analysis."""
        # Base max values
        max_values = self.max_values
        
        # Adjust based on query type
        if query_type in ["digest", "weekly_digest", "monthly_digest"]:
            return max_values.get("digest", 9999)  # No practical limit for digest requests
        elif query_type in ["analysis", "trend_analysis", "user_analysis", "content_analysis"]:
            return max_values.get("analysis", 200)
        elif analysis.get("scope") == "all_channels":
            return max_values.get("cross_server", 9999)  # No practical limit for cross-server
        else:
            return max_values.get("search", 100)
    
    def get_optimal_batch_size(self, k_value: int) -> int:
        """Get optimal batch size for processing large k values."""
        batch_size = self.performance.get("batch_size", 50)
        return min(batch_size, k_value)
    
    def should_enable_caching(self, k_value: int) -> bool:
        """Determine if caching should be enabled based on k value."""
        return self.performance.get("enable_caching", True) and k_value > 20 