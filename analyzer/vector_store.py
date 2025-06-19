"""
Vector store wrapper for ChromaDB with enhanced functionality
Provides additional utilities for managing document collections
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from django.conf import settings
from django.contrib.auth.models import User

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Enhanced vector store manager with additional utilities
    """
    
    def __init__(self, user: User):
        self.user = user
        self.chroma_client = None
        self.embedding_model = None
        self.collections = {}
        
        if CHROMADB_AVAILABLE:
            self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and embedding model"""
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
            
            logger.info(f"Vector store manager initialized for user {self.user.username}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store manager: {e}")
            self.chroma_client = None
            self.embedding_model = None
    
    def is_available(self) -> bool:
        """Check if vector store is available"""
        return (CHROMADB_AVAILABLE and 
                self.chroma_client is not None and 
                self.embedding_model is not None)
    
    def get_collection(self, collection_type: str = "documents") -> Optional[Any]:
        """
        Get or create a collection for the user
        
        Args:
            collection_type: Type of collection (documents, metadata, etc.)
        
        Returns:
            ChromaDB collection or None if not available
        """
        if not self.is_available():
            return None
        
        collection_name = f"user_{self.user.id}_{collection_type}"
        
        if collection_name in self.collections:
            return self.collections[collection_name]
        
        try:
            # Try to get existing collection
            collection = self.chroma_client.get_collection(collection_name)
        except:
            # Create new collection if it doesn't exist
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={
                    "description": f"{collection_type} for user {self.user.username}",
                    "user_id": self.user.id,
                    "created_at": datetime.now().isoformat()
                }
            )
        
        self.collections[collection_name] = collection
        return collection
    
    def batch_add_documents(self, documents: List[Dict[str, Any]], 
                           collection_type: str = "documents", 
                           batch_size: int = 100) -> bool:
        """
        Add documents in batches for better performance
        
        Args:
            documents: List of documents to add
            collection_type: Type of collection to add to
            batch_size: Number of documents per batch
        
        Returns:
            Success status
        """
        if not self.is_available() or not documents:
            return False
        
        collection = self.get_collection(collection_type)
        if not collection:
            return False
        
        try:
            total_added = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Prepare batch data
                contents = [doc['content'] for doc in batch]
                embeddings = self.embedding_model.encode(contents).tolist()
                ids = [doc['id'] for doc in batch]
                metadatas = [doc.get('metadata', {}) for doc in batch]
                
                # Add batch to collection
                collection.add(
                    embeddings=embeddings,
                    documents=contents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                total_added += len(batch)
                logger.info(f"Added batch {i//batch_size + 1}: {len(batch)} documents")
            
            logger.info(f"Successfully added {total_added} documents to {collection_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to batch add documents: {e}")
            return False
    
    def search_with_filters(self, query: str, 
                           collection_type: str = "documents",
                           n_results: int = 5,
                           doc_types: Optional[List[str]] = None,
                           date_range: Optional[Tuple[str, str]] = None,
                           client_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Advanced search with multiple filter options
        
        Args:
            query: Search query
            collection_type: Collection to search in
            n_results: Number of results to return
            doc_types: Filter by document types
            date_range: Filter by date range (start_date, end_date)
            client_names: Filter by specific client names
        
        Returns:
            List of search results
        """
        if not self.is_available():
            return []
        
        collection = self.get_collection(collection_type)
        if not collection:
            return []
        
        try:
            # Build filter conditions
            where_conditions = {}
            
            if doc_types:
                if len(doc_types) == 1:
                    where_conditions["doc_type"] = doc_types[0]
                else:
                    where_conditions["doc_type"] = {"$in": doc_types}
            
            if client_names:
                if len(client_names) == 1:
                    where_conditions["client_name"] = client_names[0]
                else:
                    where_conditions["client_name"] = {"$in": client_names}
            
            # Note: ChromaDB doesn't support date range queries directly
            # We'll filter results after retrieval if date_range is specified
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Search
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=n_results * 2 if date_range else n_results,  # Get more if we need to filter by date
                where=where_conditions if where_conditions else None
            )
            
            # Format results
            documents = []
            for i in range(len(results['ids'][0])):
                doc = {
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                
                # Apply date range filter if specified
                if date_range:
                    doc_date = doc['metadata'].get('date')
                    if doc_date and self._is_date_in_range(doc_date, date_range):
                        documents.append(doc)
                else:
                    documents.append(doc)
                
                # Stop if we have enough results
                if len(documents) >= n_results:
                    break
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to search with filters: {e}")
            return []
    
    def _is_date_in_range(self, doc_date: str, date_range: Tuple[str, str]) -> bool:
        """Check if document date is within specified range"""
        try:
            from datetime import datetime
            doc_dt = datetime.strptime(doc_date, '%Y-%m-%d')
            start_dt = datetime.strptime(date_range[0], '%Y-%m-%d')
            end_dt = datetime.strptime(date_range[1], '%Y-%m-%d')
            return start_dt <= doc_dt <= end_dt
        except:
            return True  # Include document if date parsing fails
    
    def get_document_by_id(self, doc_id: str, collection_type: str = "documents") -> Optional[Dict[str, Any]]:
        """Retrieve a specific document by ID"""
        if not self.is_available():
            return None
        
        collection = self.get_collection(collection_type)
        if not collection:
            return None
        
        try:
            results = collection.get(ids=[doc_id])
            
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document by ID: {e}")
            return None
    
    def update_document_metadata(self, doc_id: str, new_metadata: Dict[str, Any], 
                                collection_type: str = "documents") -> bool:
        """Update metadata for a specific document"""
        if not self.is_available():
            return False
        
        collection = self.get_collection(collection_type)
        if not collection:
            return False
        
        try:
            # Get current document
            current_doc = self.get_document_by_id(doc_id, collection_type)
            if not current_doc:
                return False
            
            # Merge metadata
            updated_metadata = {**current_doc['metadata'], **new_metadata}
            updated_metadata['updated_at'] = datetime.now().isoformat()
            
            # Update document
            collection.update(
                ids=[doc_id],
                metadatas=[updated_metadata]
            )
            
            logger.info(f"Updated metadata for document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document metadata: {e}")
            return False
    
    def delete_documents_by_filter(self, collection_type: str = "documents", 
                                  doc_types: Optional[List[str]] = None,
                                  client_names: Optional[List[str]] = None) -> int:
        """
        Delete documents matching filter criteria
        
        Returns:
            Number of documents deleted
        """
        if not self.is_available():
            return 0
        
        collection = self.get_collection(collection_type)
        if not collection:
            return 0
        
        try:
            # Build filter conditions
            where_conditions = {}
            
            if doc_types:
                if len(doc_types) == 1:
                    where_conditions["doc_type"] = doc_types[0]
                else:
                    where_conditions["doc_type"] = {"$in": doc_types}
            
            if client_names:
                if len(client_names) == 1:
                    where_conditions["client_name"] = client_names[0]
                else:
                    where_conditions["client_name"] = {"$in": client_names}
            
            if not where_conditions:
                logger.warning("No filter conditions provided for deletion")
                return 0
            
            # Get documents to delete
            results = collection.get(where=where_conditions)
            doc_ids = results['ids']
            
            if doc_ids:
                collection.delete(ids=doc_ids)
                logger.info(f"Deleted {len(doc_ids)} documents")
                return len(doc_ids)
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to delete documents by filter: {e}")
            return 0
    
    def get_collection_info(self, collection_type: str = "documents") -> Dict[str, Any]:
        """Get detailed information about a collection"""
        if not self.is_available():
            return {"error": "Vector store not available"}
        
        collection = self.get_collection(collection_type)
        if not collection:
            return {"error": "Collection not found"}
        
        try:
            count = collection.count()
            
            # Get sample of documents to analyze doc types
            sample_results = collection.get(limit=100)
            doc_types = {}
            client_names = set()
            
            for metadata in sample_results.get('metadatas', []):
                doc_type = metadata.get('doc_type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
                if 'client_name' in metadata:
                    client_names.add(metadata['client_name'])
            
            return {
                "collection_name": collection.name,
                "total_documents": count,
                "doc_types": doc_types,
                "unique_clients": len(client_names),
                "sample_clients": list(client_names)[:10],  # First 10 clients
                "user": self.user.username
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}
