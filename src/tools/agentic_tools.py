"""
Agentic Tools for Web Search, Function Calling, and RAG Retrieval
"""

import json
import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
from loguru import logger
from ..utils.gemini_client import get_gemini_llm


class WebSearchTool:
    """Web search capabilities using DuckDuckGo."""
    
    def __init__(self, timeout: int = 10):
        """Initialize web search tool."""
        self.timeout = timeout
    
    def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Simple web search - just get results from DDGS and return them.
        """
        try:
            logger.info(f"DDGS searching for: '{query}'")
            
            # Simple DDGS search with region for better results
            ddgs = DDGS()
            search_results = list(ddgs.text(
                query=query, 
                max_results=max_results,
                region='us-en'  # Use US English for better results
            ))
            
            logger.info(f"Raw DDGS returned {len(search_results)} results")
            
            # Process results and filter out low-quality ones
            results = []
            for i, result in enumerate(search_results):
                title = result.get("title", "")
                body = result.get("body", "")
                href = result.get("href", "")
                
                logger.info(f"Processing result {i+1}: title='{title[:50]}...', body='{body[:100]}...'")
                
                # Basic quality check
                if title and body and len(body.strip()) > 30:
                    results.append({
                        "title": title,
                        "url": href,
                        "snippet": body,
                        "content": body
                    })
                else:
                    logger.warning(f"Skipped low-quality result: title='{title[:30]}', body_len={len(body)}")
            
            logger.info(f"Returning {len(results)} quality results")
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    # Removed complex multi-backend search for simplicity
    
    def search_news(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Simple news search.
        """
        try:
            ddgs = DDGS()
            news_results = ddgs.news(query=query, max_results=max_results)
            
            results = []
            for result in news_results:
                if result.get("title") and result.get("body"):
                    results.append({
                        "title": result.get("title"),
                        "url": result.get("url", ""),
                        "snippet": result.get("body"),
                        "content": result.get("body")
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"News search error: {e}")
            return []
    
    # Removed complex content extraction and enhanced search for simplicity

class RAGRetrievalTool:
    """RAG retrieval tool for knowledge base querying."""
    
    def __init__(self):
        """Initialize RAG retrieval tool."""
        from ..database.chroma_manager import get_chroma_manager
        self.chroma_manager = get_chroma_manager()
        self.gemini_llm = get_gemini_llm()
    
    def retrieve_relevant_documents(self, 
                                  query: str,
                                  max_results: int = 5,
                                  similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from the knowledge base.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of relevant documents with metadata
        """
        try:
            logger.info(f"Retrieving documents for: {query}")
            
            # Perform similarity search
            search_results = self.chroma_manager.similarity_search(
                query=query,
                n_results=max_results
            )
            
            relevant_docs = []
            
            for i, (doc, metadata, distance) in enumerate(zip(
                search_results["documents"],
                search_results["metadatas"],
                search_results["distances"]
            )):
                similarity = 1 - distance  # Convert distance to similarity
                
                if similarity >= similarity_threshold:
                    relevant_docs.append({
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": round(similarity, 3),
                        "rank": i + 1
                    })
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def generate_contextual_response(self, 
                                   query: str,
                                   retrieved_docs: List[Dict[str, Any]],
                                   max_context_length: int = 3000) -> str:
        """
        Generate a response using retrieved documents as context.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            max_context_length: Maximum context length
            
        Returns:
            Generated response
        """
        try:
            # Build context from retrieved documents
            context_parts = []
            current_length = 0
            
            for doc in retrieved_docs:
                content = doc["content"]
                if current_length + len(content) <= max_context_length:
                    context_parts.append(f"[Source: {doc['metadata'].get('filename', 'Unknown')}]\n{content}")
                    current_length += len(content)
                else:
                    break
            
            context = "\n\n".join(context_parts)
            
            # Generate response with context
            system_instruction = """You are an intelligent assistant with access to a knowledge base. 
            Use the provided context to answer the user's question accurately and comprehensively. 
            If the context doesn't contain enough information, say so clearly."""
            
            response = self.gemini_llm.generate_response(
                prompt=query,
                context=context,
                system_instruction=system_instruction
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating contextual response: {e}")
            return "I apologize, but I encountered an error while processing your request."


class FunctionCallingTool:
    """Function calling capabilities for the agent."""
    
    def __init__(self):
        """Initialize function calling tool."""
        self.available_functions = {}
        self.register_default_functions()
    
    def register_function(self, 
                         name: str, 
                         func: Callable,
                         description: str,
                         parameters: Dict[str, Any]):
        """
        Register a new function for the agent to call.
        
        Args:
            name: Function name
            func: Function to call
            description: Function description
            parameters: Function parameters schema
        """
        self.available_functions[name] = {
            "function": func,
            "description": description,
            "parameters": parameters
        }
        logger.info(f"Registered function: {name}")
    
    def register_default_functions(self):
        """Register default utility functions."""
        
        # Calculator function
        def calculate(expression: str) -> str:
            """Calculate mathematical expressions safely."""
            try:
                # Basic safety check
                allowed_chars = "0123456789+-*/().= "
                if all(c in allowed_chars for c in expression):
                    result = eval(expression)
                    return f"Result: {result}"
                else:
                    return "Error: Invalid characters in expression"
            except Exception as e:
                return f"Error: {str(e)}"
        
        self.register_function(
            name="calculate",
            func=calculate,
            description="Calculate mathematical expressions",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to calculate"
                    }
                },
                "required": ["expression"]
            }
        )
        
        # Date/time function
        def get_current_datetime() -> str:
            """Get current date and time."""
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.register_function(
            name="get_current_datetime",
            func=get_current_datetime,
            description="Get current date and time",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    
    def call_function(self, 
                     function_name: str, 
                     parameters: Dict[str, Any]) -> str:
        """
        Call a registered function with parameters.
        
        Args:
            function_name: Name of function to call
            parameters: Function parameters
            
        Returns:
            Function result as string
        """
        try:
            if function_name not in self.available_functions:
                return f"Error: Function '{function_name}' not found"
            
            func_info = self.available_functions[function_name]
            func = func_info["function"]
            
            # Call function with parameters
            if parameters:
                result = func(**parameters)
            else:
                result = func()
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Error calling function {function_name}: {e}")
            return f"Error calling function: {str(e)}"
    
    def get_available_functions(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available functions with their descriptions."""
        return {
            name: {
                "description": info["description"],
                "parameters": info["parameters"]
            }
            for name, info in self.available_functions.items()
        }


class AgenticToolsManager:
    """Main manager for all agentic tools."""
    
    def __init__(self):
        """Initialize tools manager."""
        self.web_search = WebSearchTool()
        self.rag_retrieval = RAGRetrievalTool()
        self.function_calling = FunctionCallingTool()
        self.gemini_llm = get_gemini_llm()
    
    def process_user_query(self, 
                          query: str,
                          use_web_search: bool = True,
                          use_rag: bool = True,
                          max_web_results: int = 3,
                          max_rag_results: int = 5) -> Dict[str, Any]:
        """
        Process user query using multiple tools and generate comprehensive response.
        
        Args:
            query: User query
            use_web_search: Whether to use web search
            use_rag: Whether to use RAG retrieval
            max_web_results: Max web search results
            max_rag_results: Max RAG results
            
        Returns:
            Comprehensive response with sources
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Initialize results
            web_results = []
            rag_results = []
            
            # Enhanced web search if enabled
            if use_web_search:
                enhanced_search = self.web_search.get_enhanced_search_results(query, max_web_results)
                web_results = enhanced_search.get("results", [])
            
            # RAG retrieval if enabled
            if use_rag:
                rag_results = self.rag_retrieval.retrieve_relevant_documents(query, max_rag_results)
            
            # Combine all information for response generation
            context_parts = []
            
            # Add enhanced web search results
            if web_results:
                web_context = "Current Web Information:\n"
                for i, result in enumerate(web_results):
                    title = result.get('title', 'Unknown')
                    source = result.get('url', '')
                    
                    # Use enhanced summary if available, otherwise fall back to snippet
                    content = result.get('combined_summary', result.get('snippet', ''))
                    
                    web_context += f"{i+1}. **{title}**\n"
                    web_context += f"   Source: {source}\n"
                    web_context += f"   Content: {content}\n\n"
                context_parts.append(web_context)
            
            # Add RAG results
            if rag_results:
                rag_context = "Knowledge Base Results:\n"
                for i, result in enumerate(rag_results):
                    rag_context += f"{i+1}. {result['content'][:300]}...\n"
                context_parts.append(rag_context)
            
            # Generate comprehensive response
            full_context = "\n\n".join(context_parts)
            
            system_instruction = """You are an intelligent assistant with access to web search and a knowledge base.
            Use the provided information to give a comprehensive, accurate, and helpful response.
            Cite your sources when possible and indicate whether information comes from web search or knowledge base."""
            
            response = self.gemini_llm.generate_response(
                prompt=query,
                context=full_context,
                system_instruction=system_instruction
            )
            
            return {
                "response": response,
                "web_results": web_results,
                "rag_results": rag_results,
                "sources_used": {
                    "web_search": len(web_results) > 0,
                    "knowledge_base": len(rag_results) > 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your request.",
                "error": str(e),
                "web_results": [],
                "rag_results": [],
                "sources_used": {"web_search": False, "knowledge_base": False}
            }
    
    def get_tool_status(self) -> Dict[str, Any]:
        """Get status of all tools."""
        return {
            "web_search": "Available",
            "rag_retrieval": "Available",
            "function_calling": f"{len(self.function_calling.available_functions)} functions available",
            "available_functions": list(self.function_calling.get_available_functions().keys())
        }


# Global instance
_tools_manager = None

def get_tools_manager() -> AgenticToolsManager:
    """Get global tools manager instance."""
    global _tools_manager
    if _tools_manager is None:
        _tools_manager = AgenticToolsManager()
    return _tools_manager