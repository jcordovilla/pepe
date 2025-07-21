"""
Result Aggregator

Properly combines and synthesizes results from multiple agents.
Fixes the issue where results are scattered across different state keys.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .base_agent import BaseAgent, AgentRole, AgentState, SubTask, TaskStatus, agent_registry

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Strategies for aggregating results"""
    MERGE = "merge"
    CONCATENATE = "concatenate"
    PRIORITIZE = "prioritize"
    DEDUPLICATE = "deduplicate"
    SYNTHESIZE = "synthesize"


@dataclass
class AggregationResult:
    """Result of aggregation operation"""
    aggregated_data: Dict[str, Any]
    strategy_used: AggregationStrategy
    source_results: List[str]
    confidence_score: float
    metadata: Dict[str, Any]


class ResultAggregator(BaseAgent):
    """
    Agent responsible for combining and synthesizing results from multiple agents.
    
    This agent:
    - Combines search results from different strategies
    - Merges analysis insights from multiple agents
    - Resolves conflicts between different results
    - Creates unified, coherent responses
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.ANALYZER, config)
        
        # Aggregation configuration
        self.max_combined_results = config.get("max_combined_results", 50)
        self.deduplication_threshold = config.get("deduplication_threshold", 0.8)
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        
        # Strategy mapping for different result types
        self.strategy_mapping = {
            "search_results": AggregationStrategy.DEDUPLICATE,
            "analysis_results": AggregationStrategy.MERGE,
            "digest_results": AggregationStrategy.SYNTHESIZE,
            "insights": AggregationStrategy.MERGE,
            "summaries": AggregationStrategy.PRIORITIZE
        }
        
        logger.info(f"ResultAggregator initialized with max_combined_results={self.max_combined_results}")
        
        # Register this agent
        agent_registry.register_agent(self)
    
    def can_handle(self, task: SubTask) -> bool:
        """Determine if this agent can handle the given task."""
        if not task or not task.task_type:
            return False
        
        aggregation_types = ["aggregate_results", "combine_results", "synthesize_results"]
        task_type = task.task_type.lower() if task.task_type else ""
        return any(agg_type in task_type for agg_type in aggregation_types)
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Aggregate results from all agents into a comprehensive response.
        
        Args:
            state: Current agent state with results from other agents
            
        Returns:
            Updated state with aggregated response
        """
        try:
            # Ensure errors list exists
            if "errors" not in state or state["errors"] is None:
                state["errors"] = []
            
            logger.info("Starting result aggregation...")
            
            # Aggregate all results
            aggregated_results = await self._aggregate_all_results(state)
            
            # Generate final response
            final_response = await self._generate_final_response(
                aggregated_results, state
            )
            
            # Update state
            state["response"] = final_response
            state["metadata"]["result_aggregator"] = {
                "aggregated_results": aggregated_results,
                "final_response_length": len(final_response),
                "aggregation_time": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Result aggregation completed: {len(final_response)} chars")
            return state
            
        except Exception as e:
            logger.error(f"Error in result aggregator: {e}")
            # Ensure errors list exists before appending
            if "errors" not in state or state["errors"] is None:
                state["errors"] = []
            state["errors"].append(f"Result aggregation failed: {str(e)}")
            # Provide a fallback response
            state["response"] = "Unable to generate response due to aggregation error."
            return state
    
    async def _aggregate_all_results(self, state: AgentState) -> AggregationResult:
        """Aggregate all results from the current state"""
        search_results = state.get("search_results", []) or []
        analysis_results = state.get("analysis_results", {}) or {}
        
        # Ensure the analysis_results is a dict
        if not isinstance(analysis_results, dict):
            analysis_results = {}
        
        aggregated_data = {
            "search_results": search_results,
            "analysis_results": analysis_results,
            "total_messages": len(search_results),
            "analysis_keys": list(analysis_results.keys()),
            "aggregation_timestamp": datetime.utcnow().isoformat()
        }
        
        return AggregationResult(
            aggregated_data=aggregated_data,
            strategy_used=AggregationStrategy.SYNTHESIZE,
            source_results=["search", "analysis"],
            confidence_score=0.8,
            metadata={
                "aggregation_method": "comprehensive",
                "data_sources": ["search", "analysis"],
                "aggregated_at": datetime.utcnow().isoformat()
            }
        )
    
    async def _aggregate_search_results(self, search_results: List[Dict[str, Any]]) -> AggregationResult:
        """
        Aggregate search results from multiple sources.
        
        Args:
            search_results: List of search results
            
        Returns:
            Aggregated search results
        """
        if not search_results:
            return AggregationResult(
                aggregated_data=[],
                strategy_used=AggregationStrategy.MERGE,
                source_results=[],
                confidence_score=0.0,
                metadata={"empty": True}
            )
        
        # Deduplicate results based on content similarity
        deduplicated_results = await self._deduplicate_search_results(search_results)
        
        # Sort by relevance score if available
        sorted_results = sorted(
            deduplicated_results,
            key=lambda x: x.get("relevance_score", 0.0),
            reverse=True
        )
        
        # Limit to maximum results
        final_results = sorted_results[:self.max_combined_results]
        
        return AggregationResult(
            aggregated_data=final_results,
            strategy_used=AggregationStrategy.DEDUPLICATE,
            source_results=[f"search_{i}" for i in range(len(search_results))],
            confidence_score=min(1.0, len(final_results) / max(len(search_results), 1)),
            metadata={
                "original_count": len(search_results),
                "deduplicated_count": len(final_results),
                "max_results": self.max_combined_results
            }
        )
    
    async def _aggregate_analysis_results(self, analysis_results: Dict[str, Any]) -> AggregationResult:
        """
        Aggregate analysis results from multiple agents.
        
        Args:
            analysis_results: Dictionary of analysis results
            
        Returns:
            Aggregated analysis results
        """
        if not analysis_results:
            return AggregationResult(
                aggregated_data={},
                strategy_used=AggregationStrategy.MERGE,
                source_results=[],
                confidence_score=0.0,
                metadata={"empty": True}
            )
        
        # Merge all analysis results
        merged_results = {}
        
        for key, value in analysis_results.items():
            if isinstance(value, dict):
                # Merge dictionaries
                if key in merged_results:
                    merged_results[key].update(value)
                else:
                    merged_results[key] = value.copy()
            elif isinstance(value, list):
                # Concatenate lists
                if key in merged_results:
                    merged_results[key].extend(value)
                else:
                    merged_results[key] = value.copy()
            else:
                # Replace other types
                merged_results[key] = value
        
        return AggregationResult(
            aggregated_data=merged_results,
            strategy_used=AggregationStrategy.MERGE,
            source_results=list(analysis_results.keys()),
            confidence_score=0.9,  # High confidence for merged analysis
            metadata={
                "analysis_types": list(analysis_results.keys()),
                "merged_keys": list(merged_results.keys())
            }
        )
    
    async def _aggregate_digest_results(self, digest_results: Dict[str, Any]) -> AggregationResult:
        """
        Aggregate digest results.
        
        Args:
            digest_results: Dictionary of digest results
            
        Returns:
            Aggregated digest results
        """
        if not digest_results:
            return AggregationResult(
                aggregated_data={},
                strategy_used=AggregationStrategy.SYNTHESIZE,
                source_results=[],
                confidence_score=0.0,
                metadata={"empty": True}
            )
        
        # For digest results, we typically want to synthesize them
        synthesized_digest = await self._synthesize_digest_results(digest_results)
        
        return AggregationResult(
            aggregated_data=synthesized_digest,
            strategy_used=AggregationStrategy.SYNTHESIZE,
            source_results=list(digest_results.keys()),
            confidence_score=0.85,
            metadata={
                "digest_types": list(digest_results.keys()),
                "synthesized": True
            }
        )
    
    async def _deduplicate_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate search results based on content similarity.
        
        Args:
            results: List of search results
            
        Returns:
            Deduplicated results
        """
        if len(results) <= 1:
            return results
        
        deduplicated = []
        seen_contents = set()
        
        for result in results:
            content = result.get("content", "")
            if not content:
                deduplicated.append(result)
                continue
            
            # Simple content-based deduplication
            content_hash = hash(content.lower().strip())
            
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                deduplicated.append(result)
        
        return deduplicated
    
    async def _synthesize_digest_results(self, digest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize digest results into a coherent summary.
        
        Args:
            digest_results: Dictionary of digest results
            
        Returns:
            Synthesized digest
        """
        synthesized = {
            "summary": "",
            "key_insights": [],
            "trends": [],
            "recommendations": []
        }
        
        # Combine summaries
        summaries = []
        for key, value in digest_results.items():
            if isinstance(value, dict) and "summary" in value:
                summaries.append(value["summary"])
            elif isinstance(value, str):
                summaries.append(value)
        
        if summaries:
            synthesized["summary"] = " ".join(summaries)
        
        # Combine insights
        for key, value in digest_results.items():
            if isinstance(value, dict):
                if "insights" in value:
                    synthesized["key_insights"].extend(value["insights"])
                if "trends" in value:
                    synthesized["trends"].extend(value["trends"])
                if "recommendations" in value:
                    synthesized["recommendations"].extend(value["recommendations"])
        
        return synthesized
    
    async def _generate_final_response(self, aggregated_results: AggregationResult, state: AgentState) -> str:
        """
        Create the final response from aggregated results.
        
        Args:
            aggregated_results: Aggregated result data
            state: Current agent state
            
        Returns:
            Final response string
        """
        try:
            # Extract key information
            search_results = aggregated_results.aggregated_data.get("search_results", [])
            analysis_results = aggregated_results.aggregated_data.get("analysis_results", {})
            
            # Build response based on available data
            if search_results:
                response = f"Found {len(search_results)} relevant messages. "
                if analysis_results:
                    response += f"Analysis includes: {', '.join(analysis_results.keys())}."
                else:
                    response += "Raw search results available."
            elif analysis_results:
                response = f"Analysis completed with {len(analysis_results)} components: {', '.join(analysis_results.keys())}."
            else:
                response = "No specific results found, but query was processed successfully."
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating final response: {e}")
            return "Response generated with some processing errors." 