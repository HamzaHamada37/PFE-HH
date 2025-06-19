"""
RAG (Retrieval-Augmented Generation) Service for AI Chatbot
Integrates with ChromaDB for vector storage and retrieval
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB or sentence-transformers not available. RAG functionality will be limited.")

from django.conf import settings
from django.contrib.auth.models import User
from .models import AnalysisResult, SurveyAnalysis, ClientNote, JiraFile, SurveyFile

logger = logging.getLogger(__name__)


class RAGService:
    """
    Core RAG service for document indexing, retrieval, and context generation
    """
    
    def __init__(self, user: User):
        self.user = user
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        
        if CHROMADB_AVAILABLE:
            self._initialize_components()
    
    def _initialize_components(self):
        """Initialize ChromaDB and embedding model"""
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize ChromaDB client
            chroma_db_path = os.path.join(settings.BASE_DIR, 'chroma_db')
            os.makedirs(chroma_db_path, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=chroma_db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection for this user
            collection_name = f"user_{self.user.id}_documents"
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
            except:
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": f"Documents for user {self.user.username}"}
                )
                
            logger.info(f"RAG service initialized for user {self.user.username}")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            self.embedding_model = None
            self.chroma_client = None
            self.collection = None
    
    def is_available(self) -> bool:
        """Check if RAG service is available and properly initialized"""
        return (CHROMADB_AVAILABLE and 
                self.embedding_model is not None and 
                self.collection is not None)
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector store
        
        Args:
            documents: List of document dictionaries with keys:
                - id: unique identifier
                - content: text content to embed
                - metadata: additional metadata
        
        Returns:
            bool: Success status
        """
        if not self.is_available():
            logger.warning("RAG service not available for adding documents")
            return False
        
        try:
            # Extract content and generate embeddings
            contents = [doc['content'] for doc in documents]
            embeddings = self.embedding_model.encode(contents).tolist()
            
            # Prepare data for ChromaDB
            ids = [doc['id'] for doc in documents]
            metadatas = [doc.get('metadata', {}) for doc in documents]
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            return False
    
    def search_documents(self, query: str, n_results: int = 5, 
                        filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant documents based on query
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
        
        Returns:
            List of relevant documents with metadata and scores
        """
        if not self.is_available():
            logger.warning("RAG service not available for search")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            documents = []
            for i in range(len(results['ids'][0])):
                documents.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
            
            logger.info(f"Found {len(documents)} relevant documents for query")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
    
    def update_document(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Update an existing document in the vector store"""
        if not self.is_available():
            return False
        
        try:
            # Generate new embedding
            embedding = self.embedding_model.encode([content]).tolist()
            
            # Update in ChromaDB
            self.collection.update(
                ids=[doc_id],
                embeddings=embedding,
                documents=[content],
                metadatas=[metadata]
            )
            
            logger.info(f"Updated document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector store"""
        if not self.is_available():
            return False
        
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection"""
        if not self.is_available():
            return {"error": "RAG service not available"}
        
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection.name,
                "user": self.user.username
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection (use with caution)"""
        if not self.is_available():
            return False
        
        try:
            # Delete the collection and recreate it
            collection_name = self.collection.name
            self.chroma_client.delete_collection(collection_name)
            
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": f"Documents for user {self.user.username}"}
            )
            
            logger.info(f"Cleared collection for user {self.user.username}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False


class RAGQueryProcessor:
    """
    Processes user queries and retrieves relevant context for LLM generation
    """
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
    
    def process_query(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Enhanced query processing with hybrid search and source type filtering

        Args:
            query: User's question/query
            conversation_history: Previous conversation messages

        Returns:
            Dictionary containing retrieved context and metadata
        """
        if not self.rag_service.is_available():
            return {
                "context": "",
                "sources": [],
                "query_type": "fallback",
                "confidence": 0.0
            }

        # Determine query type and adjust search strategy
        query_type = self._classify_query(query)

        # ENHANCED: Get search filters with source type support
        search_filters = self._get_enhanced_search_filters(query_type, query)

        # ENHANCED: Try hybrid search for better client overview retrieval
        if query_type == "client_analysis" or "client" in query.lower() or "overview" in query.lower():
            relevant_docs = self._search_with_hybrid_fallback(
                query=query,
                n_results=7,  # Get more results for client queries
                filter_metadata=search_filters
            )
        else:
            # Standard search for other query types
            relevant_docs = self.rag_service.search_documents(
                query=query,
                n_results=5,
                filter_metadata=search_filters
            )

        # Build context from retrieved documents
        context = self._build_context(relevant_docs, query_type)

        return {
            "context": context,
            "sources": [doc['metadata'] for doc in relevant_docs],
            "query_type": query_type,
            "confidence": self._calculate_confidence(relevant_docs),
            "search_method": "hybrid" if query_type == "client_analysis" else "semantic",
            "total_docs_found": len(relevant_docs)
        }
    
    def _classify_query(self, query: str) -> str:
        """Enhanced query classification to optimize retrieval for client overview data"""
        query_lower = query.lower()

        # ENHANCED: More specific client overview detection
        if any(phrase in query_lower for phrase in ['client overview', 'client data', 'client metrics', 'client performance']):
            return "client_analysis"
        elif any(word in query_lower for word in ['client', 'customer']) and any(word in query_lower for word in ['sentiment', 'resolution', 'tickets', 'impact']):
            return "client_analysis"
        elif any(word in query_lower for word in ['client', 'customer']):
            return "client_analysis"
        elif any(word in query_lower for word in ['team', 'survey', 'satisfaction']):
            return "team_analysis"
        elif any(word in query_lower for word in ['trend', 'performance', 'metric']):
            return "performance_analysis"
        elif any(word in query_lower for word in ['note', 'comment', 'feedback']):
            return "qualitative_analysis"
        else:
            return "general"
    
    def _get_search_filters(self, query_type: str) -> Optional[Dict]:
        """Get metadata filters based on query type with enhanced client overview support"""
        filters = {
            "client_analysis": {
                "doc_type": {"$in": ["client_overview", "client_overview_summary", "client_data", "analysis_summary", "business_insights"]}
            },
            "team_analysis": {
                "doc_type": {"$in": ["survey_summary", "team_kpis", "team_feedback"]}
            },
            "performance_analysis": {
                "doc_type": {"$in": ["client_overview", "client_data", "team_kpis", "analysis_summary"]}
            },
            "qualitative_analysis": {
                "doc_type": {"$in": ["client_notes", "team_feedback"]}
            },
            "general": None
        }
        return filters.get(query_type)

    def _get_enhanced_search_filters(self, query_type: str, query: str) -> Optional[Dict]:
        """Enhanced search filters with source type and keyword-based filtering"""
        base_filters = self._get_search_filters(query_type)

        # Add source type filters for specific queries
        if "client" in query.lower() and "overview" in query.lower():
            return {
                "$or": [
                    {"source_type": "client_analysis"},
                    {"source_type": "aggregated_analysis"},
                    {"doc_type": {"$in": ["client_overview", "client_overview_summary"]}}
                ]
            }
        elif "client" in query.lower():
            return {
                "$or": [
                    {"source_type": "client_analysis"},
                    {"doc_type": {"$in": ["client_overview", "client_data", "analysis_summary"]}}
                ]
            }

        return base_filters
    
    def _build_context(self, documents: List[Dict], query_type: str) -> str:
        """Build context string from retrieved documents"""
        if not documents:
            return ""
        
        context_parts = []
        for doc in documents:
            metadata = doc.get('metadata', {})
            content = doc.get('content', '')
            
            # Add source information
            source_info = f"Source: {metadata.get('source', 'Unknown')}"
            if metadata.get('date'):
                source_info += f" (Date: {metadata['date']})"
            
            context_parts.append(f"{source_info}\n{content}\n")
        
        return "\n---\n".join(context_parts)
    
    def _calculate_confidence(self, documents: List[Dict]) -> float:
        """Calculate confidence score based on retrieval results"""
        if not documents:
            return 0.0

        # Get distance scores
        distances = [doc.get('distance', 2.0) for doc in documents if doc.get('distance') is not None]
        if not distances:
            return 0.7  # Default confidence when no distance info but documents found

        # Convert distance to similarity (lower distance = higher similarity)
        # Typical distances range from 0.5 to 2.0, so we normalize accordingly
        avg_distance = sum(distances) / len(distances)

        # Improved confidence calculation
        if avg_distance <= 0.8:
            confidence = 0.9  # Very high confidence
        elif avg_distance <= 1.2:
            confidence = 0.7  # High confidence
        elif avg_distance <= 1.6:
            confidence = 0.5  # Medium confidence
        else:
            confidence = 0.3  # Low confidence

        return confidence

    def _search_with_hybrid_fallback(self, query: str, n_results: int = 5,
                                    filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Hybrid search with semantic + keyword fallback for better client overview retrieval
        """
        if not self.rag_service.is_available():
            return []

        # First try: Standard semantic search
        semantic_results = self.rag_service.search_documents(
            query=query,
            n_results=n_results,
            filter_metadata=filter_metadata
        )

        # If semantic search returns good results, use them
        if semantic_results and len(semantic_results) >= n_results // 2:
            avg_distance = sum(doc.get('distance', 1.0) for doc in semantic_results) / len(semantic_results)
            if avg_distance <= 1.5:  # Good semantic match
                return semantic_results

        # Fallback: Try broader search without filters
        if filter_metadata:
            broader_results = self.rag_service.search_documents(
                query=query,
                n_results=n_results * 2,
                filter_metadata=None  # Remove filters for broader search
            )

            # Filter results manually for client-related content
            client_results = []
            for doc in broader_results:
                content = doc.get('content', '').lower()
                metadata = doc.get('metadata', {})

                # Check if document contains client-related content
                if (any(keyword in content for keyword in ['client', 'customer', 'overview', 'sentiment', 'resolution', 'tickets']) or
                    metadata.get('doc_type') in ['client_overview', 'client_data', 'analysis_summary']):
                    client_results.append(doc)

                if len(client_results) >= n_results:
                    break

            if client_results:
                return client_results

        # Return semantic results even if not perfect
        return semantic_results

    def search_with_hybrid_fallback(self, query: str, n_results: int = 5,
                                   filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Enhanced search with hybrid keyword + vector fallback
        """
        if not self.rag_service.is_available():
            return []

        # First try: Standard semantic search
        semantic_results = self.rag_service.search_documents(
            query=query,
            n_results=n_results,
            filter_metadata=filter_metadata
        )

        # If semantic search returns good results, use them
        if semantic_results and len(semantic_results) >= n_results // 2:
            avg_distance = sum(doc.get('distance', 1.0) for doc in semantic_results) / len(semantic_results)
            if avg_distance <= 1.5:  # Good semantic match
                return semantic_results

        # Fallback: Keyword-based search for client overview queries
        if any(keyword in query.lower() for keyword in ['client', 'overview', 'customer']):
            keyword_results = self._keyword_search_fallback(query, n_results, filter_metadata)

            # Combine and deduplicate results
            combined_results = self._combine_search_results(semantic_results, keyword_results, n_results)
            return combined_results

        return semantic_results

    def _keyword_search_fallback(self, query: str, n_results: int,
                                filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Keyword-based fallback search for when semantic search fails"""
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query)

            # Search for documents containing keywords
            all_docs = self.rag_service.collection.get(
                where=filter_metadata,
                include=["documents", "metadatas"]
            )

            # Score documents based on keyword matches
            scored_docs = []
            for i, doc_content in enumerate(all_docs['documents']):
                score = self._calculate_keyword_score(doc_content, keywords)
                if score > 0:
                    scored_docs.append({
                        'id': all_docs['ids'][i],
                        'content': doc_content,
                        'metadata': all_docs['metadatas'][i],
                        'distance': 1.0 - score,  # Convert score to distance
                        'keyword_score': score
                    })

            # Sort by keyword score and return top results
            scored_docs.sort(key=lambda x: x['keyword_score'], reverse=True)
            return scored_docs[:n_results]

        except Exception as e:
            logger.error(f"Keyword search fallback failed: {e}")
            return []

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from query"""
        # Simple keyword extraction - can be enhanced with NLP
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'how', 'show', 'me', 'my', 'our'}
        words = query.lower().split()
        keywords = [word.strip('.,!?') for word in words if word not in stop_words and len(word) > 2]
        return keywords

    def _calculate_keyword_score(self, content: str, keywords: List[str]) -> float:
        """Calculate keyword match score for a document"""
        content_lower = content.lower()
        matches = sum(1 for keyword in keywords if keyword in content_lower)
        return matches / len(keywords) if keywords else 0

    def _combine_search_results(self, semantic_results: List[Dict], keyword_results: List[Dict],
                               n_results: int) -> List[Dict[str, Any]]:
        """Combine and deduplicate semantic and keyword search results"""
        seen_ids = set()
        combined = []

        # Add semantic results first (higher priority)
        for doc in semantic_results:
            if doc['id'] not in seen_ids:
                doc['search_type'] = 'semantic'
                combined.append(doc)
                seen_ids.add(doc['id'])

        # Add keyword results that weren't already included
        for doc in keyword_results:
            if doc['id'] not in seen_ids and len(combined) < n_results:
                doc['search_type'] = 'keyword'
                combined.append(doc)
                seen_ids.add(doc['id'])

        return combined[:n_results]
