"""
ChromaDB Vector Database Manager
Handles document storage, indexing, and retrieval with embeddings
"""

import os
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from loguru import logger
import numpy as np
from sentence_transformers import SentenceTransformer


class ChromaDBManager:
    """Advanced ChromaDB manager for vector storage and retrieval."""
    
    def __init__(self, 
                 db_path: str = "./data/chromadb",
                 collection_name: str = "knowledge_base",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.7):
        """
        Initialize ChromaDB manager.
        
        Args:
            db_path: Path to ChromaDB storage
            collection_name: Name of the collection
            embedding_model: Sentence transformer model for embeddings
            similarity_threshold: Minimum similarity score for retrieval
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        
        # Ensure directory exists
        os.makedirs(db_path, exist_ok=True)
        
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")
            
            # Create or get collection
            self.collection = self._get_or_create_collection()
            
            logger.info(f"Initialized ChromaDB at {db_path} with collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            self.client = None
            self.collection = None
    
    def _get_or_create_collection(self):
        """Get existing collection or create a new one."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Found existing collection: {self.collection_name}")
            return collection
            
        except Exception:
            # Create new collection
            logger.info(f"Creating new collection: {self.collection_name}")
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"created_at": datetime.now().isoformat()}
            )
    
    def add_documents(self, 
                     documents: List[str],
                     metadatas: List[Dict[str, Any]],
                     document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            document_ids: Optional list of document IDs
            
        Returns:
            Result with success status and statistics
        """
        try:
            if not self._is_available():
                return {"success": False, "error": "ChromaDB not available"}
            
            if not documents:
                return {"success": False, "error": "No documents provided"}
            
            # Generate IDs if not provided
            if document_ids is None:
                document_ids = [str(uuid.uuid4()) for _ in documents]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Add timestamp to metadata
            for metadata in metadatas:
                metadata["added_at"] = datetime.now().isoformat()
            
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=document_ids
            )
            
            result = {
                "success": True,
                "documents_added": len(documents),
                "ids": document_ids,
                "collection_size": self.collection.count()
            }
            
            logger.info(f"Added {len(documents)} documents to ChromaDB")
            return result
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            return {"success": False, "error": str(e)}
    
    def add_processed_document(self, processed_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a processed document with its chunks to the database.
        
        Args:
            processed_doc: Document processed by DocumentProcessor
            
        Returns:
            Result with success status
        """
        try:
            if not processed_doc.get("success", False):
                return {"success": False, "error": "Document processing failed"}
            
            chunks = processed_doc.get("chunks", [])
            if not chunks:
                return {"success": False, "error": "No chunks found in processed document"}
            
            # Create metadata for each chunk
            base_metadata = {
                "filename": processed_doc.get("filename", "unknown"),
                "file_type": processed_doc.get("file_type", "unknown"),
                "document_hash": processed_doc.get("document_hash", ""),
                "processed_at": processed_doc.get("processed_at", datetime.now().isoformat()),
                "total_chunks": len(chunks)
            }
            
            # Add document-level metadata
            doc_metadata = processed_doc.get("metadata", {})
            base_metadata.update(doc_metadata)
            
            # Prepare chunk data
            chunk_documents = []
            chunk_metadatas = []
            chunk_ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "chunk_id": f"{processed_doc.get('document_hash', 'unknown')}_{i}",
                    "chunk_size": len(chunk)
                })
                
                chunk_documents.append(chunk)
                chunk_metadatas.append(chunk_metadata)
                chunk_ids.append(chunk_metadata["chunk_id"])
            
            # Add to database
            return self.add_documents(
                documents=chunk_documents,
                metadatas=chunk_metadatas,
                document_ids=chunk_ids
            )
            
        except Exception as e:
            logger.error(f"Error adding processed document: {e}")
            return {"success": False, "error": str(e)}
    
    def similarity_search(self, 
                         query: str,
                         n_results: int = 5,
                         where: Optional[Dict] = None,
                         include_distances: bool = True) -> Dict[str, Any]:
        """
        Perform similarity search in the vector database.
        
        Args:
            query: Search query
            n_results: Number of results to return
            where: Optional metadata filter
            include_distances: Whether to include similarity distances
            
        Returns:
            Search results with documents, metadata, and distances
        """
        try:
            if not self._is_available():
                return {"documents": [], "metadatas": [], "distances": []}
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Prepare include list
            include_list = ["documents", "metadatas"]
            if include_distances:
                include_list.append("distances")
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=include_list
            )
            
            # Format results
            search_results = {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if include_distances and results.get("distances") else [],
                "total_results": len(results["documents"][0]) if results["documents"] else 0
            }
            
            logger.info(f"Similarity search returned {search_results['total_results']} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return {"documents": [], "metadatas": [], "distances": [], "error": str(e)}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            if not self._is_available():
                return {"error": "ChromaDB not available"}
            
            count = self.collection.count()
            
            # Get sample metadata to analyze
            sample_docs = self.collection.get(
                limit=min(100, count),
                include=["metadatas"]
            )
            
            # Analyze file types
            file_types = {}
            unique_files = set()
            
            for metadata in sample_docs.get("metadatas", []):
                file_type = metadata.get("file_type", "unknown")
                filename = metadata.get("filename", "unknown")
                
                file_types[file_type] = file_types.get(file_type, 0) + 1
                unique_files.add(filename)
            
            return {
                "total_documents": len(unique_files),
                "total_chunks": count,
                "unique_files": len(unique_files),
                "file_types": file_types,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def reset_collection(self) -> bool:
        """Reset (clear) the entire collection."""
        try:
            if not self._is_available():
                return False
            
            # Delete the collection
            self.client.delete_collection(name=self.collection_name)
            
            # Recreate it
            self.collection = self._get_or_create_collection()
            
            logger.info(f"Reset collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False
    
    def _is_available(self) -> bool:
        """Check if ChromaDB is available."""
        return self.client is not None and self.collection is not None
    
    def force_recovery(self) -> bool:
        """Attempt to recover from database errors."""
        try:
            logger.info("Attempting ChromaDB recovery...")
            
            # Reinitialize client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            
            # Try to get collection
            self.collection = self._get_or_create_collection()
            
            logger.info("ChromaDB recovery successful")
            return True
            
        except Exception as e:
            logger.error(f"ChromaDB recovery failed: {e}")
            return False


# Global instance
_chroma_manager = None

def get_chroma_manager() -> ChromaDBManager:
    """Get global ChromaDB manager instance."""
    global _chroma_manager
    if _chroma_manager is None:
        from config.settings import config
        _chroma_manager = ChromaDBManager(
            db_path=config.chroma_db_path,
            collection_name=config.collection_name
        )
    return _chroma_manager