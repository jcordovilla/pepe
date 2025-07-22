"""
Search Agent

Specialized agent for vector search, semantic retrieval, and message filtering.
"""

from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
import asyncio
import dateutil.parser

from .base_agent import BaseAgent, AgentRole, AgentState, SubTask, TaskStatus, agent_registry
from ..vectorstore.persistent_store import PersistentVectorStore
from ..cache.smart_cache import SmartCache

logger = logging.getLogger(__name__)


class SearchAgent(BaseAgent):
    """
    Agent responsible for all search-related operations.
    
    This agent:
    - Performs semantic vector search using ChromaDB
    - Handles keyword-based filtering
    - Manages search result ranking and reranking
    - Implements smart caching for performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.SEARCHER, config)
        
        # Initialize vector store and cache
        self.vector_store = PersistentVectorStore(config.get("vectorstore", {}))
        self.cache = SmartCache(config.get("cache", {}))
        
        # Search configuration
        self.default_k = config.get("default_k", 10)
        self.max_k = config.get("max_k", 100)
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        self.use_reranking = config.get("use_reranking", True)
        
        # Performance tracking
        self.search_stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info(f"SearchAgent initialized with default_k={self.default_k}")
        
        # Register this agent
        agent_registry.register_agent(self)
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process search-related subtasks.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with search results
        """
        try:
            # Prefer current_subtask if present (orchestrator mode)
            if "current_subtask" in state and state["current_subtask"] is not None:
                subtask = state["current_subtask"]
                if self.can_handle(subtask):
                    logger.info(f"Processing search subtask: {subtask.task_type}")
                    if subtask.task_type == "semantic_search" or subtask.task_type == "search":
                        task_results = await self._semantic_search(subtask, state)
                    elif subtask.task_type == "keyword_search":
                        task_results = await self._keyword_search(subtask, state)
                    elif subtask.task_type == "filtered_search":
                        task_results = await self._filtered_search(subtask, state)
                    elif subtask.task_type == "hybrid_search":
                        task_results = await self._hybrid_search(subtask, state)
                    elif subtask.task_type == "reaction_search":
                        task_results = await self._reaction_search(subtask, state)
                    else:
                        logger.warning(f"Unknown search task type: {subtask.task_type}")
                        task_results = []
                    
                    # Update state and subtask
                    state["search_results"] = task_results
                    subtask.status = TaskStatus.COMPLETED
                    subtask.result = task_results
                    state["metadata"]["search_agent"] = {
                        "search_time": datetime.utcnow().isoformat(),
                        "total_results": len(task_results),
                        "subtasks_processed": 1,
                        "stats": self.search_stats.copy()
                    }
                    logger.info(f"Search completed: {len(task_results)} results found")
                    return state
                else:
                    logger.warning("Current subtask is not a search subtask")
                    return state
            # Fallback: process all subtasks if present (batch mode)
            subtasks = state.get("subtasks", [])
            if subtasks is None:
                subtasks = []
            
            search_subtasks = [task for task in subtasks if self.can_handle(task)]
            if not search_subtasks:
                logger.warning("No search subtasks found")
                return state
            
            results = []
            for subtask in search_subtasks:
                logger.info(f"Processing search subtask: {subtask.task_type}")
                task_results = []
                
                try:
                    if subtask.task_type == "semantic_search" or subtask.task_type == "search":
                        task_results = await self._semantic_search(subtask, state)
                    elif subtask.task_type == "keyword_search":
                        task_results = await self._keyword_search(subtask, state)
                    elif subtask.task_type == "filtered_search":
                        task_results = await self._filtered_search(subtask, state)
                    elif subtask.task_type == "hybrid_search":
                        task_results = await self._hybrid_search(subtask, state)
                    elif subtask.task_type == "reaction_search":
                        task_results = await self._reaction_search(subtask, state)
                    else:
                        logger.warning(f"Unknown search task type: {subtask.task_type}")
                        continue
                except Exception as e:
                    logger.error(f"Error processing search subtask {subtask.task_type}: {e}")
                    task_results = []
                
                # Ensure task_results is a list
                if task_results is None:
                    task_results = []
                
                results.extend(task_results)
                subtask.status = TaskStatus.COMPLETED
                subtask.result = task_results
            
            # Deduplicate and rank results
            if results:
                results = await self._deduplicate_results(results)
                results = await self._rank_results(results, state)
            
            state["search_results"] = results
            state["metadata"]["search_agent"] = {
                "search_time": datetime.utcnow().isoformat(),
                "total_results": len(results),
                "subtasks_processed": len(search_subtasks),
                "stats": self.search_stats.copy()
            }
            logger.info(f"Search completed: {len(results)} results found")
            return state
        except Exception as e:
            logger.error(f"Error in search agent: {e}")
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Search error: {str(e)}")
            return state
    
    def can_handle(self, task: SubTask) -> bool:
        """
        Determine if this agent can handle the given task.
        
        Args:
            task: Task to evaluate
            
        Returns:
            True if task is search-related
        """
        if not task or not task.task_type:
            return False
            
        search_types = [
            "search", "semantic_search", "keyword_search", 
            "filtered_search", "hybrid_search", "vector_search",
            "reaction_search"  # Add reaction search support
        ]
        task_type = task.task_type.lower() if task.task_type else ""
        return any(search_type in task_type for search_type in search_types)
    
    async def _semantic_search(self, subtask: SubTask, state: AgentState) -> List[Dict[str, Any]]:
        """
        Perform semantic vector search.
        
        Args:
            subtask: Search subtask
            state: Current agent state
            
        Returns:
            List of search results
        """
        try:
            query = subtask.parameters.get("query", state.get("query", ""))
            if not query:
                logger.warning("No query provided for semantic search")
                return []
            
            query = str(query)  # Ensure query is a string
            k = subtask.parameters.get("k", subtask.parameters.get("limit", self.default_k))
            
            # Check cache first
            cache_key = f"semantic:{hash(query)}:{k}:{hash(str(subtask.parameters.get('filters', {})))}"
            cached_results = await self.cache.get(cache_key)
            
            if cached_results:
                self.search_stats["cache_hits"] += 1
                logger.info("Semantic search cache hit")
                return cached_results if isinstance(cached_results, list) else []
            
            # Perform vector search
            self.search_stats["cache_misses"] += 1
            self.search_stats["total_searches"] += 1
            
            logger.info(f"Performing semantic search: query='{query[:50]}...', k={k}, filters={subtask.parameters.get('filters', {})}")
            
            results = await self.vector_store.similarity_search(
                query=query,
                k=min(k, self.max_k),
                filters=subtask.parameters.get("filters", {})
            )
            
            # Ensure results is a list
            if results is None:
                results = []
            
            # Cache results
            await self.cache.set(cache_key, results, ttl=3600)  # 1 hour TTL
            
            logger.info(f"Semantic search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    async def _keyword_search(self, subtask: SubTask, state: AgentState) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search.
        
        Args:
            subtask: Search subtask
            state: Current agent state
            
        Returns:
            List of search results
        """
        try:
            keywords = subtask.parameters.get("keywords", [])
            k = subtask.parameters.get("k", self.default_k)
            
            if not keywords:
                return []
            
            # Check cache
            cache_key = f"keyword:{hash(str(keywords))}:{k}"
            cached_results = await self.cache.get(cache_key)
            
            if cached_results:
                self.search_stats["cache_hits"] += 1
                return cached_results
            
            # Perform keyword search
            self.search_stats["cache_misses"] += 1
            self.search_stats["total_searches"] += 1
            
            results = await self.vector_store.keyword_search(
                keywords=keywords,
                k=min(k, self.max_k),
                filters=subtask.parameters.get("filters", {})
            )
            
            # Cache results
            await self.cache.set(cache_key, results, ttl=3600)
            
            logger.info(f"Keyword search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    async def _filtered_search(self, subtask: SubTask, state: AgentState) -> List[Dict[str, Any]]:
        """
        Perform filtered search with specific criteria.
        
        Args:
            subtask: Search subtask
            state: Current agent state
            
        Returns:
            List of search results
        """
        try:
            filters = subtask.parameters.get("filters", {})
            k = subtask.parameters.get("k", self.default_k)
            sort_by = subtask.parameters.get("sort_by", "timestamp")  # Get sort parameter
            
            # Build comprehensive filters from entities
            # Get entities from query interpretation (list format)
            query_interpretation = state.get("query_interpretation", {})
            entities_list = query_interpretation.get("entities", [])
            
            # Convert entities list to organized dict for easier processing
            entities = {}
            for entity in entities_list:
                entity_type = entity.get("type")
                entity_value = entity.get("value")
                
                if entity_type == "channel":
                    if "channels" not in entities:
                        entities["channels"] = []
                    entities["channels"].append(entity_value)
                elif entity_type == "time_range":
                    entities["time_range"] = entity
                elif entity_type == "user":
                    if "authors" not in entities:
                        entities["authors"] = []
                    entities["authors"].append(entity_value)
            
            # Clean up conflicting filters from subtask parameters
            # Remove any conflicting filters that might be in the subtask parameters
            filters.pop("channel_id", None)  # Remove if present
            filters.pop("time_range", None)  # Remove if present
            
            # Build ChromaDB-compatible where clause with $and operator for multiple conditions
            where_conditions = []
            
            # Add channel filters
            if "channels" in entities and "channel_id" not in filters:
                where_conditions.append({"channel_id": {"$in": entities["channels"]}})
            
            # Add time filters
            if "time_range" in entities and "timestamp_unix" not in filters:
                time_range = entities["time_range"]
                if time_range.get("start"):
                    try:
                        start_dt = dateutil.parser.isoparse(time_range.get("start"))
                        start_ts = start_dt.timestamp()
                        where_conditions.append({
                            "timestamp_unix": {"$gte": start_ts}
                        })
                    except Exception as e:
                        print(f"[DEBUG] Failed to parse start timestamp: {e}")
                if time_range.get("end"):
                    try:
                        end_dt = dateutil.parser.isoparse(time_range.get("end"))
                        end_ts = end_dt.timestamp()
                        where_conditions.append({
                            "timestamp_unix": {"$lte": end_ts}
                        })
                    except Exception as e:
                        print(f"[DEBUG] Failed to parse end timestamp: {e}")
            
            # Add author filters
            if "authors" in entities and "author.username" not in filters:
                where_conditions.append({"author.username": {"$in": entities["authors"]}})
            
            # Combine conditions with $and operator if multiple conditions
            if len(where_conditions) == 1:
                filters = where_conditions[0]
            elif len(where_conditions) > 1:
                filters = {"$and": where_conditions}
            
            logger.info(f"Performing filtered search with sort_by='{sort_by}', filters={filters}")
            print(f"[DEBUG] Final filters passed to vector store: {filters}")
            
            # Perform filtered search
            if subtask.parameters.get("query") and not sort_by:
                # Semantic search with filters (only if no temporal sorting needed)
                results = await self.vector_store.similarity_search(
                    query=subtask.parameters["query"],
                    k=min(k, self.max_k),
                    filters=filters
                )
            else:
                # Pure filter-based search with sorting (for temporal queries)
                results = await self.vector_store.filter_search(
                    filters=filters,
                    k=min(k, self.max_k),
                    sort_by=sort_by
                )
            
            logger.info(f"Filtered search found {len(results)} results sorted by {sort_by}")
            return results
            
        except Exception as e:
            logger.error(f"Error in filtered search: {e}")
            return []
    
    async def _hybrid_search(self, subtask: SubTask, state: AgentState) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            subtask: Search subtask
            state: Current agent state
            
        Returns:
            List of search results
        """
        try:
            query = subtask.parameters.get("query", "")
            keywords = subtask.parameters.get("keywords", [])
            k = subtask.parameters.get("k", self.default_k)
            
            # Perform both searches concurrently
            semantic_task = self._semantic_search(
                SubTask(
                    id=f"{subtask.id}_semantic",
                    description="Semantic search for hybrid",
                    agent_role=AgentRole.SEARCHER,
                    task_type="semantic_search",
                    parameters={"query": query, "k": k//2},
                    dependencies=[]
                ),
                state
            )
            
            keyword_task = self._keyword_search(
                SubTask(
                    id=f"{subtask.id}_keyword",
                    description="Keyword search for hybrid", 
                    agent_role=AgentRole.SEARCHER,
                    task_type="keyword_search",
                    parameters={"keywords": keywords, "k": k//2},
                    dependencies=[]
                ),
                state
            )
            
            semantic_results, keyword_results = await asyncio.gather(
                semantic_task, keyword_task
            )
            
            # Combine and deduplicate results
            combined_results = semantic_results + keyword_results
            combined_results = await self._deduplicate_results(combined_results)
            
            # Rerank based on hybrid scoring
            if self.use_reranking:
                combined_results = await self._hybrid_rerank(
                    combined_results, query, keywords
                )
            
            # Limit to requested number
            combined_results = combined_results[:k]
            
            logger.info(f"Hybrid search found {len(combined_results)} results")
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    async def _reaction_search(self, subtask: SubTask, state: AgentState) -> List[Dict[str, Any]]:
        """
        Perform search based on message reactions (likes, loves, etc.).
        
        Args:
            subtask: Search subtask
            state: Current agent state
            
        Returns:
            List of search results
        """
        try:
            reaction_type = subtask.parameters.get("reaction", "")
            k = subtask.parameters.get("k", self.default_k)
            
            if not reaction_type:
                return []
            
            # Check cache
            cache_key = f"reaction:{hash(reaction_type)}:{k}"
            cached_results = await self.cache.get(cache_key)
            
            if cached_results:
                self.search_stats["cache_hits"] += 1
                return cached_results
            
            # Perform reaction-based search
            self.search_stats["cache_misses"] += 1
            self.search_stats["total_searches"] += 1
            
            results = await self.vector_store.reaction_search(
                reaction=reaction_type,
                k=min(k, self.max_k),
                filters=subtask.parameters.get("filters", {})
            )
            
            # Cache results
            await self.cache.set(cache_key, results, ttl=3600)
            
            logger.info(f"Reaction search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in reaction search: {e}")
            return []
    
    async def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate results based on message ID.
        
        Args:
            results: List of search results
            
        Returns:
            Deduplicated results
        """
        seen_ids = set()
        deduplicated = []
        
        for result in results:
            message_id = result.get("message_id") or result.get("id")
            if message_id and message_id not in seen_ids:
                seen_ids.add(message_id)
                deduplicated.append(result)
        
        return deduplicated
    
    async def _rank_results(self, results: List[Dict[str, Any]], state: AgentState) -> List[Dict[str, Any]]:
        """
        Rank search results based on relevance and recency.
        
        Args:
            results: Search results to rank
            state: Current agent state
            
        Returns:
            Ranked results
        """
        try:
            if not results:
                return results
            
            # Simple ranking based on similarity score and recency
            for result in results:
                score = result.get("score", 0.5)
                
                # Boost recent messages
                timestamp = result.get("timestamp", "")
                if timestamp:
                    try:
                        msg_date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        days_old = (datetime.now().replace(tzinfo=msg_date.tzinfo) - msg_date).days
                        recency_boost = max(0, 1 - (days_old / 30))  # Boost for messages < 30 days old
                        score += recency_boost * 0.1
                    except:
                        pass
                
                result["final_score"] = score
            
            # Sort by final score
            results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error ranking results: {e}")
            return results
    
    async def _hybrid_rerank(self, results: List[Dict[str, Any]], query: str, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Rerank results using hybrid scoring.
        
        Args:
            results: Results to rerank
            query: Original query
            keywords: Keywords from query
            
        Returns:
            Reranked results
        """
        try:
            for result in results:
                content = result.get("content", "").lower()
                
                # Semantic score (from vector search)
                semantic_score = result.get("score", 0.5)
                
                # Keyword score
                keyword_score = 0
                if keywords:
                    keyword_matches = sum(1 for kw in keywords if kw.lower() in content)
                    keyword_score = keyword_matches / len(keywords)
                
                # Combined score
                hybrid_score = (semantic_score * 0.7) + (keyword_score * 0.3)
                result["hybrid_score"] = hybrid_score
            
            # Sort by hybrid score
            results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid reranking: {e}")
            return results
    
    def get_search_stats(self) -> Dict[str, Any]:
        """
        Get search performance statistics.
        
        Returns:
            Dictionary containing search stats
        """
        total = self.search_stats["total_searches"]
        cache_rate = (
            self.search_stats["cache_hits"] / total * 100 
            if total > 0 else 0
        )
        
        return {
            **self.search_stats,
            "cache_hit_rate": f"{cache_rate:.1f}%",
            "agent_type": "search",
            "role": self.role.value
        }
