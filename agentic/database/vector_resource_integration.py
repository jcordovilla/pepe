"""
Vector Store Resource Integration

Unified integration layer between ChromaDB vector store and enhanced resource database
for seamless search, retrieval, and resource management.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

import chromadb
from chromadb.config import Settings

from .enhanced_resource_database import EnhancedResourceDatabase, ResourceRecord
from ..services.enhanced_resource_detector import ResourceMetadata
from ..services.resource_validation_pipeline import ValidationResult

logger = logging.getLogger(__name__)


class VectorResourceIntegration:
    """
    Unified integration between vector store and resource database
    """
    
    def __init__(self, chromadb_path: str = "data/chromadb", resource_db_path: str = "data/enhanced_resources.db"):
        self.chromadb_path = chromadb_path
        self.resource_db_path = resource_db_path
        
        # Initialize components
        self.chroma_client = None
        self.collection = None
        self.resource_db = None
        
        # Cache for performance
        self.message_to_embedding_cache = {}
        self.resource_cache = {}
        
        logger.info("🔗 Vector-Resource integration initialized")
    
    async def initialize(self):
        """Initialize both vector store and resource database"""
        try:
            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=self.chromadb_path,
                settings=Settings(allow_reset=True)
            )
            
            # Get main collection
            collections = self.chroma_client.list_collections()
            if collections:
                self.collection = collections[0]  # Use first collection (likely discord_messages)
                logger.info(f"📚 Connected to ChromaDB collection: {self.collection.name}")
            else:
                logger.warning("⚠️ No ChromaDB collections found")
            
            # Initialize resource database
            self.resource_db = EnhancedResourceDatabase(self.resource_db_path)
            await self.resource_db.initialize()
            
            logger.info("✅ Vector-Resource integration ready")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Vector-Resource integration: {e}")
            raise
    
    async def store_message_with_resources(
        self, 
        message: Dict[str, Any], 
        resources: List[ResourceMetadata], 
        validations: List[ValidationResult],
        embedding_id: Optional[str] = None
    ) -> bool:
        """
        Store message resources with vector store integration
        """
        try:
            # Get or create embedding ID
            if not embedding_id:
                embedding_id = await self._get_embedding_id_for_message(message.get('message_id'))
            
            # Store each resource with validation
            success_count = 0
            for resource, validation in zip(resources, validations):
                try:
                    # Store in resource database with embedding link
                    stored = await self.resource_db.store_resource(resource, validation, embedding_id)
                    if stored:
                        success_count += 1
                        
                        # Update cache
                        self.resource_cache[resource.id] = {
                            'resource': resource,
                            'validation': validation,
                            'embedding_id': embedding_id
                        }
                    
                except Exception as e:
                    logger.error(f"❌ Error storing resource {resource.id}: {e}")
                    continue
            
            logger.info(f"✅ Stored {success_count}/{len(resources)} resources for message {message.get('message_id')}")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ Error in store_message_with_resources: {e}")
            return False
    
    async def search_resources_with_context(
        self, 
        query: str, 
        resource_types: Optional[List[str]] = None,
        quality_threshold: float = 0.5,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search resources using both vector similarity and metadata filtering
        """
        try:
            results = []
            
            # 1. Vector similarity search in ChromaDB for context
            if self.collection:
                vector_results = self.collection.query(
                    query_texts=[query],
                    n_results=limit * 2,  # Get more to filter
                    include=['documents', 'metadatas', 'distances']
                )
                
                # Extract message IDs from vector results
                relevant_message_ids = []
                if vector_results and vector_results.get('metadatas'):
                    for metadata in vector_results['metadatas'][0]:
                        if isinstance(metadata, dict) and 'message_id' in metadata:
                            relevant_message_ids.append(metadata['message_id'])
            else:
                relevant_message_ids = []
            
            # 2. Get resources from database with filtering
            if relevant_message_ids:
                # Build SQL query for message-based filtering
                placeholders = ','.join(['?' for _ in relevant_message_ids])
                base_query = f"""
                    SELECT r.*, 
                           COUNT(ua.id) as usage_count,
                           MAX(ua.accessed_at) as last_access
                    FROM resources r
                    LEFT JOIN resource_usage_analytics ua ON r.id = ua.resource_id
                    WHERE r.message_id IN ({placeholders})
                      AND r.is_active = TRUE
                      AND r.quality_score >= ?
                """
                
                params = relevant_message_ids + [quality_threshold]
                
                # Add resource type filtering
                if resource_types:
                    type_placeholders = ','.join(['?' for _ in resource_types])
                    base_query += f" AND r.resource_type IN ({type_placeholders})"
                    params.extend(resource_types)
                
                base_query += """
                    GROUP BY r.id
                    ORDER BY r.quality_score DESC, usage_count DESC, r.created_at DESC
                    LIMIT ?
                """
                params.append(limit)
                
                cursor = self.resource_db.connection.execute(base_query, params)
                db_results = [dict(row) for row in cursor.fetchall()]
                
                # Enhance with vector context
                for resource in db_results:
                    # Find corresponding vector result for relevance scoring
                    relevance_score = 0.5  # Default
                    context_snippet = ""
                    
                    if vector_results and vector_results.get('documents'):
                        for i, metadata in enumerate(vector_results['metadatas'][0]):
                            if (isinstance(metadata, dict) and 
                                metadata.get('message_id') == resource['message_id']):
                                
                                # Calculate relevance from vector distance
                                if i < len(vector_results['distances'][0]):
                                    distance = vector_results['distances'][0][i]
                                    relevance_score = max(0, 1 - distance)  # Convert distance to similarity
                                
                                # Get context snippet
                                if i < len(vector_results['documents'][0]):
                                    context_snippet = vector_results['documents'][0][i][:200]
                                
                                break
                    
                    # Enhance resource data
                    enhanced_resource = {
                        **resource,
                        'relevance_score': relevance_score,
                        'context_snippet': context_snippet,
                        'search_query': query
                    }
                    
                    results.append(enhanced_resource)
            
            else:
                # Fallback: Direct database search if no vector results
                results = await self._fallback_resource_search(query, resource_types, quality_threshold, limit)
            
            # Sort by combined relevance and quality score
            results.sort(
                key=lambda x: (x.get('relevance_score', 0) * 0.6 + x['quality_score'] * 0.4),
                reverse=True
            )
            
            # Track search analytics
            await self._track_search_analytics(query, len(results), resource_types)
            
            logger.info(f"🔍 Found {len(results)} resources for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in search_resources_with_context: {e}")
            return []
    
    async def get_resource_recommendations(
        self, 
        user_id: str, 
        context: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get personalized resource recommendations based on user behavior
        """
        try:
            # Get user's recent resource interactions
            cursor = self.resource_db.connection.execute("""
                SELECT r.resource_type, r.domain, COUNT(*) as interaction_count
                FROM resource_usage_analytics ua
                JOIN resources r ON ua.resource_id = r.id
                WHERE ua.user_id = ? 
                  AND ua.accessed_at >= datetime('now', '-30 days')
                GROUP BY r.resource_type, r.domain
                ORDER BY interaction_count DESC
                LIMIT 5
            """, (user_id,))
            
            user_preferences = [dict(row) for row in cursor.fetchall()]
            
            # Get trending high-quality resources
            cursor = self.resource_db.connection.execute("""
                SELECT r.*, COUNT(ua.id) as recent_access_count
                FROM resources r
                LEFT JOIN resource_usage_analytics ua ON r.id = ua.resource_id 
                  AND ua.accessed_at >= datetime('now', '-7 days')
                WHERE r.quality_score >= 0.7
                  AND r.is_active = TRUE
                  AND r.validation_status IN ('healthy', 'degraded')
                GROUP BY r.id
                ORDER BY recent_access_count DESC, r.quality_score DESC
                LIMIT ?
            """, (limit * 2,))
            
            trending_resources = [dict(row) for row in cursor.fetchall()]
            
            # Apply user preference scoring
            scored_resources = []
            for resource in trending_resources:
                base_score = resource['quality_score']
                preference_boost = 0
                
                # Boost based on user preferences
                for pref in user_preferences:
                    if resource['resource_type'] == pref['resource_type']:
                        preference_boost += 0.2
                    if resource['domain'] == pref['domain']:
                        preference_boost += 0.1
                
                # Context relevance boost
                context_boost = 0
                if context and resource.get('description'):
                    # Simple keyword overlap for context relevance
                    context_words = set(context.lower().split())
                    desc_words = set(resource['description'].lower().split())
                    overlap = len(context_words.intersection(desc_words))
                    context_boost = min(0.2, overlap * 0.02)
                
                final_score = base_score + preference_boost + context_boost
                
                scored_resources.append({
                    **resource,
                    'recommendation_score': final_score,
                    'preference_boost': preference_boost,
                    'context_boost': context_boost
                })
            
            # Sort by recommendation score and limit
            scored_resources.sort(key=lambda x: x['recommendation_score'], reverse=True)
            recommendations = scored_resources[:limit]
            
            # Track recommendation analytics
            await self._track_recommendation_analytics(user_id, len(recommendations), context)
            
            logger.info(f"💡 Generated {len(recommendations)} recommendations for user {user_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
            return []
    
    async def get_resource_clusters(self, min_cluster_size: int = 3) -> List[Dict[str, Any]]:
        """
        Get resource clusters for topic analysis and navigation
        """
        try:
            # Cluster by domain and resource type
            cursor = self.resource_db.connection.execute("""
                SELECT 
                    domain,
                    resource_type,
                    COUNT(*) as resource_count,
                    AVG(quality_score) as avg_quality,
                    MIN(created_at) as first_seen,
                    MAX(created_at) as last_seen,
                    GROUP_CONCAT(id, '|') as resource_ids
                FROM resources
                WHERE is_active = TRUE 
                  AND domain IS NOT NULL
                GROUP BY domain, resource_type
                HAVING COUNT(*) >= ?
                ORDER BY resource_count DESC, avg_quality DESC
            """, (min_cluster_size,))
            
            domain_clusters = [dict(row) for row in cursor.fetchall()]
            
            # Cluster by content similarity (simplified by keywords/topics)
            cursor = self.resource_db.connection.execute("""
                SELECT 
                    rm.value as topic,
                    COUNT(DISTINCT r.id) as resource_count,
                    AVG(r.quality_score) as avg_quality,
                    GROUP_CONCAT(r.id, '|') as resource_ids
                FROM resource_metadata rm
                JOIN resources r ON rm.resource_id = r.id
                WHERE rm.key = 'topics' 
                  AND r.is_active = TRUE
                GROUP BY rm.value
                HAVING COUNT(DISTINCT r.id) >= ?
                ORDER BY resource_count DESC
            """, (min_cluster_size,))
            
            topic_clusters = [dict(row) for row in cursor.fetchall()]
            
            # Combine and format clusters
            clusters = []
            
            # Add domain clusters
            for cluster in domain_clusters:
                clusters.append({
                    'type': 'domain',
                    'name': f"{cluster['domain']} - {cluster['resource_type']}",
                    'resource_count': cluster['resource_count'],
                    'avg_quality': cluster['avg_quality'],
                    'cluster_data': cluster
                })
            
            # Add topic clusters
            for cluster in topic_clusters:
                clusters.append({
                    'type': 'topic',
                    'name': f"Topic: {cluster['topic']}",
                    'resource_count': cluster['resource_count'],
                    'avg_quality': cluster['avg_quality'],
                    'cluster_data': cluster
                })
            
            logger.info(f"📊 Found {len(clusters)} resource clusters")
            return clusters
            
        except Exception as e:
            logger.error(f"❌ Error getting resource clusters: {e}")
            return []
    
    async def _get_embedding_id_for_message(self, message_id: str) -> Optional[str]:
        """Get embedding ID for a message from ChromaDB"""
        try:
            if not self.collection:
                return None
                
            # Query ChromaDB for message
            results = self.collection.get(
                where={"message_id": message_id},
                include=['metadatas']
            )
            
            if results and results.get('ids'):
                return results['ids'][0]
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not find embedding for message {message_id}: {e}")
            return None
    
    async def _fallback_resource_search(
        self, 
        query: str, 
        resource_types: Optional[List[str]], 
        quality_threshold: float, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Fallback search using only database text matching"""
        try:
            # Simple text search in titles and descriptions
            search_term = f"%{query}%"
            base_query = """
                SELECT r.*, 0.5 as relevance_score, '' as context_snippet
                FROM resources r
                WHERE r.is_active = TRUE
                  AND r.quality_score >= ?
                  AND (r.title LIKE ? OR r.description LIKE ? OR r.url LIKE ?)
            """
            
            params = [quality_threshold, search_term, search_term, search_term]
            
            if resource_types:
                type_placeholders = ','.join(['?' for _ in resource_types])
                base_query += f" AND r.resource_type IN ({type_placeholders})"
                params.extend(resource_types)
            
            base_query += " ORDER BY r.quality_score DESC LIMIT ?"
            params.append(limit)
            
            cursor = self.resource_db.connection.execute(base_query, params)
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"❌ Error in fallback search: {e}")
            return []
    
    async def _track_search_analytics(self, query: str, result_count: int, resource_types: Optional[List[str]]):
        """Track search analytics for optimization"""
        try:
            # This could be enhanced to integrate with the main analytics system
            logger.info(f"📈 Search: '{query[:30]}...' -> {result_count} results")
        except Exception as e:
            logger.debug(f"Error tracking search analytics: {e}")
    
    async def _track_recommendation_analytics(self, user_id: str, recommendation_count: int, context: Optional[str]):
        """Track recommendation analytics"""
        try:
            logger.info(f"💡 Recommendations for {user_id}: {recommendation_count} items")
        except Exception as e:
            logger.debug(f"Error tracking recommendation analytics: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.resource_db:
            await self.resource_db.close()
        
        # Clear caches
        self.message_to_embedding_cache.clear()
        self.resource_cache.clear() 