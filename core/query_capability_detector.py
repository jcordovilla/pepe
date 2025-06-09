"""
Query Capability Detection System

Determines the appropriate capability type for test queries based on
content analysis and pattern matching.
"""

import re
from typing import Dict, List, Optional, Any

class QueryCapabilityDetector:
    """Detects query capability based on content analysis."""
    
    def __init__(self):
        self.capability_patterns = {
            "server_data_analysis": [
                "analyze.*message.*activity",
                "analyze.*patterns",
                "message.*activity.*patterns",
                "engagement.*across.*groups",
                "activity.*patterns.*across",
                "most.*active.*channels",
                "data.*analysis",
                "statistics.*analysis",
                "activity.*distribution",
                "message.*distribution",
                "user.*engagement",
                "community.*activity"
            ],
            
            "feedback_summarization": [
                "summarize.*feedback",
                "feedback.*summary",
                "suggestions.*made",
                "member.*experiences",
                "community.*feedback",
                "feedback.*about",
                "suggestions.*about",
                "experiences.*shared",
                "feedback.*on.*workshops",
                "workshop.*feedback"
            ],
            
            "trending_topics": [
                "trending.*topics",
                "trending.*discussions",
                "popular.*topics",
                "what.*trending",
                "emerging.*patterns",
                "collaboration.*patterns",
                "project.*ideas",
                "evolution.*discussions",
                "trending.*methodologies",
                "hot.*topics"
            ],
            
            "qa_concepts": [
                "questions.*about",
                "compile.*questions",
                "frequently.*asked",
                "q.*a.*about",
                "questions.*and.*answers",
                "extract.*questions",
                "solutions.*shared",
                "questions.*solutions",
                "most.*frequently.*asked"
            ],
            
            "statistics_generation": [
                "generate.*statistics", 
                "engagement.*statistics",
                "statistics.*for",
                "provide.*statistics",
                "calculate.*response.*times",
                "interaction.*rates",
                "onboarding.*statistics",
                "member.*statistics",
                "top.*10.*most.*active",
                "statistics.*on.*new.*member"
            ],
            
            "server_structure_analysis": [
                "analyze.*channel.*utilization",
                "channel.*utilization",
                "suggest.*which.*buddy.*groups",
                "information.*flow.*between",
                "communication.*gaps",
                "recommend.*best.*practices",
                "channel.*engagement",
                "server.*structure",
                "buddy.*groups.*benefit",
                "merging.*or.*splitting"
            ]
        }
    
    def detect_capability(self, query: str) -> str:
        """
        Detect the most likely capability for a given query.
        
        Args:
            query: The user query string
            
        Returns:
            The detected capability name
        """
        query_lower = query.lower()
        
        # Score each capability based on pattern matches
        capability_scores = {}
        
        for capability, patterns in self.capability_patterns.items():
            score = 0
            
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 2  # High weight for exact pattern match
                    
            # Additional scoring based on keywords
            capability_keywords = self._get_capability_keywords(capability)
            for keyword in capability_keywords:
                if keyword in query_lower:
                    score += 1
                    
            capability_scores[capability] = score
        
        # Return the capability with the highest score
        if capability_scores:
            best_capability = max(capability_scores.items(), key=lambda x: x[1])
            if best_capability[1] > 0:
                return best_capability[0]
        
        # Default fallback
        return "general"
    
    def _get_capability_keywords(self, capability: str) -> List[str]:
        """Get keywords associated with each capability."""
        keyword_map = {
            "server_data_analysis": [
                "analyze", "analysis", "patterns", "activity", "engagement", 
                "statistics", "data", "distribution", "active", "channels"
            ],
            "feedback_summarization": [
                "feedback", "summary", "summarize", "suggestions", "experiences",
                "opinions", "thoughts", "reviews", "comments"
            ],
            "trending_topics": [
                "trending", "popular", "hot", "emerging", "topics", "discussions",
                "patterns", "collaboration", "evolution", "methodologies"
            ],
            "qa_concepts": [
                "questions", "answers", "qa", "q&a", "frequently", "asked",
                "solutions", "help", "support", "problems"
            ],
            "statistics_generation": [
                "statistics", "stats", "metrics", "numbers", "count", "rate",
                "times", "engagement", "top", "most", "calculate"
            ],
            "server_structure_analysis": [
                "structure", "organization", "channels", "utilization", "flow",
                "gaps", "recommend", "practices", "buddy", "groups", "optimize"
            ]
        }
        
        return keyword_map.get(capability, [])
    
    def get_capability_description(self, capability: str) -> str:
        """Get a human-readable description of the capability."""
        descriptions = {
            "server_data_analysis": "Analysis of server data, message patterns, and user engagement",
            "feedback_summarization": "Summarization of community feedback and member experiences", 
            "trending_topics": "Identification of trending topics and emerging patterns",
            "qa_concepts": "Compilation and analysis of questions and answers",
            "statistics_generation": "Generation of engagement and activity statistics",
            "server_structure_analysis": "Analysis of server structure and optimization recommendations",
            "general": "General query not fitting specific capability categories"
        }
        
        return descriptions.get(capability, "Unknown capability")
    
    def analyze_query_for_testing(self, query: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of a query for testing purposes.
        
        Args:
            query: The user query string
            
        Returns:
            Dictionary with analysis results
        """
        detected_capability = self.detect_capability(query)
        
        # Score all capabilities for comparison
        query_lower = query.lower()
        all_scores = {}
        
        for capability, patterns in self.capability_patterns.items():
            score = 0
            matched_patterns = []
            
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 2
                    matched_patterns.append(pattern)
                    
            capability_keywords = self._get_capability_keywords(capability)
            matched_keywords = [kw for kw in capability_keywords if kw in query_lower]
            score += len(matched_keywords)
            
            all_scores[capability] = {
                "score": score,
                "matched_patterns": matched_patterns,
                "matched_keywords": matched_keywords
            }
        
        return {
            "query": query,
            "detected_capability": detected_capability,
            "capability_description": self.get_capability_description(detected_capability),
            "all_scores": all_scores,
            "confidence": all_scores[detected_capability]["score"] / max(1, max(s["score"] for s in all_scores.values()))
        }
