"""
LangGraph-based Agentic Workflow for RAG System
Orchestrates agent actions including query analysis, retrieval, web search, and response synthesis
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Generator
from datetime import datetime
from dataclasses import dataclass
import operator

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from loguru import logger

# Import our components
from ..utils.gemini_client import get_gemini_llm
from ..utils.memory_manager import get_memory_manager
from ..utils.rlhf_manager import get_rlhf_manager
from ..tools.agentic_tools import get_tools_manager
from ..database.chroma_manager import get_chroma_manager


class AgentState(TypedDict):
    """State object for the agent workflow."""
    messages: Annotated[list, operator.add]
    session_id: str
    user_id: Optional[str]
    query: str
    query_type: str
    needs_web_search: bool
    needs_rag_retrieval: bool
    needs_function_calling: bool
    web_results: List[Dict[str, Any]]
    rag_results: List[Dict[str, Any]]
    function_results: List[Dict[str, Any]]
    context: str
    response: str
    confidence_score: float
    sources_used: Dict[str, bool]
    processing_steps: List[str]
    error: Optional[str]


@dataclass
class QueryAnalysis:
    """Result of query analysis."""
    query_type: str
    intent: str
    needs_web_search: bool
    needs_rag_retrieval: bool
    needs_function_calling: bool
    confidence: float
    reasoning: str


class AgenticWorkflow:
    """Main agentic workflow using LangGraph."""
    
    def __init__(self):
        """Initialize the agentic workflow."""
        # Initialize components
        self.llm = get_gemini_llm()
        self.memory_manager = get_memory_manager()
        self.rlhf_manager = get_rlhf_manager()
        self.tools_manager = get_tools_manager()
        self.chroma_manager = get_chroma_manager()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        logger.info("Initialized AgenticWorkflow with LangGraph")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("web_search", self._web_search)
        workflow.add_node("rag_retrieval", self._rag_retrieval)
        workflow.add_node("function_calling", self._function_calling)
        workflow.add_node("synthesize_response", self._synthesize_response)
        workflow.add_node("update_memory", self._update_memory)
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze_query",
            self._route_after_analysis,
            {
                "web_search": "web_search",
                "rag_retrieval": "rag_retrieval", 
                "function_calling": "function_calling",
                "synthesize": "synthesize_response"
            }
        )
        
        workflow.add_conditional_edges(
            "web_search",
            self._route_after_web_search,
            {
                "rag_retrieval": "rag_retrieval",
                "function_calling": "function_calling", 
                "synthesize": "synthesize_response"
            }
        )
        
        workflow.add_conditional_edges(
            "rag_retrieval",
            self._route_after_rag,
            {
                "function_calling": "function_calling",
                "synthesize": "synthesize_response"
            }
        )
        
        workflow.add_edge("function_calling", "synthesize_response")
        workflow.add_edge("synthesize_response", "update_memory")
        workflow.add_edge("update_memory", END)
        
        return workflow.compile()
    
    async def process_query(self, 
                           query: str, 
                           session_id: str,
                           user_id: Optional[str] = None,
                           use_memory: bool = True) -> Dict[str, Any]:
        """
        Process a user query through the agentic workflow.
        
        Args:
            query: User query
            session_id: Session identifier
            user_id: Optional user identifier
            use_memory: Whether to use conversation memory
            
        Returns:
            Processing result with response and metadata
        """
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Initialize state
            initial_state = AgentState(
                messages=[],
                session_id=session_id,
                user_id=user_id,
                query=query,
                query_type="unknown",
                needs_web_search=False,
                needs_rag_retrieval=False,
                needs_function_calling=False,
                web_results=[],
                rag_results=[],
                function_results=[],
                context="",
                response="",
                confidence_score=0.5,
                sources_used={
                    "web_search": False,
                    "knowledge_base": False,
                    "function_calls": False,
                    "memory": False
                },
                processing_steps=[],
                error=None
            )
            
            # Load conversation history if using memory
            if use_memory:
                history = self.memory_manager.get_session_history(session_id, limit=10)
                for msg in history[-5:]:  # Last 5 messages for context
                    if msg["role"] == "user":
                        initial_state["messages"].append(HumanMessage(content=msg["content"]))
                    else:
                        initial_state["messages"].append(AIMessage(content=msg["content"]))
                
                if history:
                    initial_state["sources_used"]["memory"] = True
            
            # Add current user message
            initial_state["messages"].append(HumanMessage(content=query))
            
            # Run the workflow
            result = await self.workflow.ainvoke(initial_state)
            
            # Prepare response
            response_data = {
                "response": result["response"],
                "confidence_score": result["confidence_score"],
                "sources_used": result["sources_used"],
                "processing_steps": result["processing_steps"],
                "web_results": result["web_results"],
                "rag_results": result["rag_results"],
                "function_results": result["function_results"],
                "query_type": result["query_type"],
                "session_id": session_id,
                "processing_time": 0.0,  # TODO: Add timing
                "error": result.get("error")
            }
            
            logger.info("Query processing completed successfully")
            return response_data
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": f"I apologize, but I encountered an error processing your request: {str(e)}",
                "confidence_score": 0.1,
                "sources_used": {"error": True},
                "processing_steps": ["error_occurred"],
                "error": str(e)
            }
    
    def stream_response(self, query: str, session_id: str) -> Generator[str, None, None]:
        """
        Stream response generation (simplified version).
        
        Args:
            query: User query
            session_id: Session identifier
            
        Yields:
            Response chunks
        """
        try:
            # For streaming, we'll use a simpler approach
            # Get conversation history
            history = self.memory_manager.get_session_history(session_id, limit=5)
            
            # Build context
            context_parts = []
            
            # Add recent conversation
            if history:
                recent_context = "Recent conversation:\n"
                for msg in history[-3:]:
                    role = "You" if msg["role"] == "user" else "Assistant"
                    recent_context += f"{role}: {msg['content'][:200]}\n"
                context_parts.append(recent_context)
            
            # Quick RAG search
            if self.chroma_manager._is_available():
                rag_results = self.chroma_manager.similarity_search(query, n_results=3)
                if rag_results["documents"]:
                    rag_context = "Knowledge base:\n"
                    for i, doc in enumerate(rag_results["documents"][:2]):
                        rag_context += f"- {doc[:200]}...\n"
                    context_parts.append(rag_context)
            
            # Quick web search
            try:
                web_results = self.tools_manager.web_search.search_web(query, max_results=2)
                if web_results:
                    web_context = "Current information:\n"
                    for result in web_results:
                        web_context += f"- {result['title']}: {result['snippet'][:150]}...\n"
                    context_parts.append(web_context)
            except:
                pass  # Web search failed, continue without it
            
            full_context = "\n\n".join(context_parts)
            
            system_instruction = """You are a helpful AI assistant. Use the provided context to give accurate, comprehensive responses. Be concise but informative."""
            
            # Stream the response
            for chunk in self.llm.stream_response(query, full_context, system_instruction):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield f"Error: {str(e)}"
    
    def _analyze_query(self, state: AgentState) -> AgentState:
        """Analyze the user query to determine required actions."""
        try:
            state["processing_steps"].append("query_analysis")
            
            query = state["query"]
            
            analysis_prompt = f"""
            Analyze this user query and determine what actions are needed:
            
            Query: "{query}"
            
            Determine:
            1. Query type (question, request, calculation, etc.)
            2. Whether web search is needed (for current events, latest info)
            3. Whether knowledge base search is needed (for stored documents)
            4. Whether function calling is needed (for calculations, dates, etc.)
            
            Respond in JSON format:
            {{
                "query_type": "question|request|calculation|other",
                "intent": "brief description",
                "needs_web_search": true/false,
                "needs_rag_retrieval": true/false, 
                "needs_function_calling": true/false,
                "confidence": 0.0-1.0,
                "reasoning": "explanation"
            }}
            """
            
            try:
                analysis_result = self.llm.generate_response(analysis_prompt)
                
                # Parse JSON response
                import re
                json_match = re.search(r'\{.*\}', analysis_result, re.DOTALL)
                if json_match:
                    analysis_data = json.loads(json_match.group())
                else:
                    # Fallback analysis
                    analysis_data = self._fallback_query_analysis(query)
                
            except:
                # Fallback analysis
                analysis_data = self._fallback_query_analysis(query)
            
            # Update state
            state["query_type"] = analysis_data.get("query_type", "question")
            state["needs_web_search"] = analysis_data.get("needs_web_search", False)
            state["needs_rag_retrieval"] = analysis_data.get("needs_rag_retrieval", True)
            state["needs_function_calling"] = analysis_data.get("needs_function_calling", False)
            
            logger.info(f"Query analysis: type={state['query_type']}, web={state['needs_web_search']}, rag={state['needs_rag_retrieval']}, func={state['needs_function_calling']}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            state["error"] = str(e)
            # Set default analysis
            state["query_type"] = "question"
            state["needs_rag_retrieval"] = True
            return state
    
    def _fallback_query_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback query analysis using simple heuristics."""
        query_lower = query.lower()
        
        # Check for calculations
        calc_keywords = ["calculate", "compute", "what is", "how much", "+", "-", "*", "/"]
        needs_calculation = any(keyword in query_lower for keyword in calc_keywords)
        
        # Check for current events/web search
        web_keywords = ["latest", "current", "news", "recent", "today", "now", "2024", "2025"]
        needs_web = any(keyword in query_lower for keyword in web_keywords)
        
        # Check for time/date
        time_keywords = ["time", "date", "when", "today", "now"]
        needs_time = any(keyword in query_lower for keyword in time_keywords)
        
        return {
            "query_type": "calculation" if needs_calculation else "question",
            "intent": "User query analysis",
            "needs_web_search": needs_web,
            "needs_rag_retrieval": True,  # Default to true
            "needs_function_calling": needs_calculation or needs_time,
            "confidence": 0.7,
            "reasoning": "Fallback heuristic analysis"
        }
    
    def _web_search(self, state: AgentState) -> AgentState:
        """Perform web search."""
        try:
            state["processing_steps"].append("web_search")
            
            query = state["query"]
            results = self.tools_manager.web_search.search_web(query, max_results=3)
            
            state["web_results"] = results
            if results:
                state["sources_used"]["web_search"] = True
            
            logger.info(f"Web search completed: {len(results)} results")
            return state
            
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            state["web_results"] = []
            return state
    
    def _rag_retrieval(self, state: AgentState) -> AgentState:
        """Perform RAG retrieval from knowledge base."""
        try:
            state["processing_steps"].append("rag_retrieval")
            
            query = state["query"]
            
            if self.chroma_manager._is_available():
                results = self.chroma_manager.similarity_search(query, n_results=5)
                
                # Format results
                rag_results = []
                for i, (doc, metadata, distance) in enumerate(zip(
                    results.get("documents", []),
                    results.get("metadatas", []),
                    results.get("distances", [])
                )):
                    similarity = 1 - distance if distance else 0.5
                    rag_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": round(similarity, 3),
                        "rank": i + 1
                    })
                
                state["rag_results"] = rag_results
                if rag_results:
                    state["sources_used"]["knowledge_base"] = True
            else:
                state["rag_results"] = []
            
            logger.info(f"RAG retrieval completed: {len(state['rag_results'])} results")
            return state
            
        except Exception as e:
            logger.error(f"Error in RAG retrieval: {e}")
            state["rag_results"] = []
            return state
    
    def _function_calling(self, state: AgentState) -> AgentState:
        """Perform function calling."""
        try:
            state["processing_steps"].append("function_calling")
            
            query = state["query"]
            query_lower = query.lower()
            
            function_results = []
            
            # Check for calculations
            if any(op in query for op in ["+", "-", "*", "/", "calculate", "compute"]):
                # Extract mathematical expression
                import re
                math_pattern = r'[\d\+\-\*/\(\)\.\s]+'
                matches = re.findall(math_pattern, query)
                
                for match in matches:
                    if any(op in match for op in ["+", "-", "*", "/"]):
                        result = self.tools_manager.function_calling.call_function(
                            "calculate", {"expression": match.strip()}
                        )
                        function_results.append({
                            "function": "calculate",
                            "input": match.strip(),
                            "result": result
                        })
            
            # Check for date/time requests
            if any(word in query_lower for word in ["time", "date", "when", "now", "today"]):
                result = self.tools_manager.function_calling.call_function(
                    "get_current_datetime", {}
                )
                function_results.append({
                    "function": "get_current_datetime",
                    "input": "current time request",
                    "result": result
                })
            
            state["function_results"] = function_results
            if function_results:
                state["sources_used"]["function_calls"] = True
            
            logger.info(f"Function calling completed: {len(function_results)} results")
            return state
            
        except Exception as e:
            logger.error(f"Error in function calling: {e}")
            state["function_results"] = []
            return state
    
    def _synthesize_response(self, state: AgentState) -> AgentState:
        """Synthesize final response from all sources."""
        try:
            state["processing_steps"].append("response_synthesis")
            
            # Build context from all sources
            context_parts = []
            
            # Add web search results
            if state["web_results"]:
                web_context = "Recent web information:\n"
                for result in state["web_results"]:
                    web_context += f"- {result['title']}: {result['snippet']}\n"
                context_parts.append(web_context)
            
            # Add RAG results
            if state["rag_results"]:
                rag_context = "Knowledge base information:\n"
                for result in state["rag_results"][:3]:  # Top 3 results
                    filename = result["metadata"].get("filename", "Unknown")
                    rag_context += f"- From {filename}: {result['content'][:300]}...\n"
                context_parts.append(rag_context)
            
            # Add function results
            if state["function_results"]:
                func_context = "Calculations and functions:\n"
                for result in state["function_results"]:
                    func_context += f"- {result['function']}: {result['result']}\n"
                context_parts.append(func_context)
            
            # Add conversation memory context
            memory_context = ""
            if len(state["messages"]) > 1:  # More than just current message
                memory_context = "Conversation context:\n"
                for msg in state["messages"][-4:-1]:  # Recent messages except current
                    role = "You" if isinstance(msg, HumanMessage) else "Assistant"
                    memory_context += f"{role}: {msg.content[:100]}...\n"
                context_parts.append(memory_context)
            
            full_context = "\n\n".join(context_parts)
            
            # Generate response
            system_instruction = f"""
            You are a helpful AI assistant. Use the provided information to answer the user's question comprehensively and accurately.
            
            Guidelines:
            - Be direct and helpful
            - Cite sources when possible (web, knowledge base, calculations)
            - If you used multiple sources, mention that
            - Be concise but thorough
            - If information is conflicting, acknowledge it
            """
            
            response = self.llm.generate_response(
                state["query"],
                full_context,
                system_instruction
            )
            
            # Calculate confidence based on sources used
            confidence = 0.5  # Base confidence
            if state["sources_used"]["web_search"]:
                confidence += 0.15
            if state["sources_used"]["knowledge_base"]:
                confidence += 0.20
            if state["sources_used"]["function_calls"]:
                confidence += 0.10
            if state["sources_used"]["memory"]:
                confidence += 0.05
            
            confidence = min(confidence, 0.95)  # Cap at 95%
            
            state["response"] = response
            state["confidence_score"] = confidence
            state["context"] = full_context
            
            logger.info(f"Response synthesis completed, confidence: {confidence:.2f}")
            return state
            
        except Exception as e:
            logger.error(f"Error synthesizing response: {e}")
            state["response"] = f"I apologize, but I encountered an error generating a response: {str(e)}"
            state["confidence_score"] = 0.1
            state["error"] = str(e)
            return state
    
    def _update_memory(self, state: AgentState) -> AgentState:
        """Update conversation memory."""
        try:
            state["processing_steps"].append("memory_update")
            
            session_id = state["session_id"]
            user_id = state.get("user_id")
            
            # Add user message to memory
            self.memory_manager.add_message(
                session_id=session_id,
                role="user",
                content=state["query"],
                user_id=user_id,
                metadata={
                    "query_type": state["query_type"],
                    "sources_requested": {
                        "web_search": state["needs_web_search"],
                        "rag_retrieval": state["needs_rag_retrieval"],
                        "function_calling": state["needs_function_calling"]
                    }
                }
            )
            
            # Add assistant response to memory
            self.memory_manager.add_message(
                session_id=session_id,
                role="assistant",
                content=state["response"],
                user_id=user_id,
                metadata={
                    "confidence_score": state["confidence_score"],
                    "sources_used": state["sources_used"],
                    "processing_steps": state["processing_steps"],
                    "web_results_count": len(state["web_results"]),
                    "rag_results_count": len(state["rag_results"]),
                    "function_results_count": len(state["function_results"])
                }
            )
            
            logger.info("Memory updated successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            return state
    
    def _route_after_analysis(self, state: AgentState) -> str:
        """Route to appropriate node after query analysis."""
        if state["needs_web_search"]:
            return "web_search"
        elif state["needs_rag_retrieval"]:
            return "rag_retrieval"
        elif state["needs_function_calling"]:
            return "function_calling"
        else:
            return "synthesize"
    
    def _route_after_web_search(self, state: AgentState) -> str:
        """Route after web search."""
        if state["needs_rag_retrieval"]:
            return "rag_retrieval"
        elif state["needs_function_calling"]:
            return "function_calling"
        else:
            return "synthesize"
    
    def _route_after_rag(self, state: AgentState) -> str:
        """Route after RAG retrieval."""
        if state["needs_function_calling"]:
            return "function_calling"
        else:
            return "synthesize"
    
    # Convenience methods for external access
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        return self.memory_manager.get_session_history(session_id)
    
    def clear_memory(self, session_id: str) -> bool:
        """Clear conversation memory for a session."""
        return self.memory_manager.clear_session(session_id)
    
    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get user context and profile."""
        return self.memory_manager.get_user_context(user_id)


# Global instance
_agentic_workflow = None

def get_agentic_workflow() -> AgenticWorkflow:
    """Get global agentic workflow instance."""
    global _agentic_workflow
    if _agentic_workflow is None:
        _agentic_workflow = AgenticWorkflow()
    return _agentic_workflow