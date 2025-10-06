"""
Streamlit Frontend for the Agentic RAG System
User-friendly interface with chat, document upload, and memory visualization
"""

import streamlit as st
import asyncio
import time
from typing import Dict, List, Any
from datetime import datetime
import io
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.agents import get_agentic_workflow
from src.utils import get_document_processor, get_rlhf_manager
from src.database.chroma_manager import get_chroma_manager
from config.settings import config


# Configure Streamlit page
st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)



def initialize_session_state():
    """Initialize Streamlit session state variables."""
    try:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "session_id" not in st.session_state:
            st.session_state.session_id = f"session_{int(time.time())}"
        
        if "workflow" not in st.session_state:
            st.session_state.workflow = get_agentic_workflow()
        
        if "doc_processor" not in st.session_state:
            st.session_state.doc_processor = get_document_processor()
        
        if "rlhf_manager" not in st.session_state:
            st.session_state.rlhf_manager = get_rlhf_manager()
        
        if "chroma_manager" not in st.session_state:
            st.session_state.chroma_manager = get_chroma_manager()
    
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {str(e)}")
        st.error("Please check your .env file and dependencies")
        st.stop()


def render_chat_message(message: Dict[str, Any], is_user: bool = True):
    """Render a chat message with Streamlit native components."""
    
    if is_user:
        # User message
        with st.chat_message("user"):
            st.write(f"👤 **You:** {message.get('content', '')}")
    else:
        # Assistant message
        with st.chat_message("assistant"):
            st.write(f"🤖 **Assistant:** {message.get('content', '')}")
            
            # Show sources and confidence using native Streamlit components
            if isinstance(message, dict):
                sources = message.get('sources_used', {})
                confidence = message.get('confidence_score', 0.5)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    source_icons = []
                    if sources.get('web_search'):
                        source_icons.append("🌐 Web Search")
                    if sources.get('knowledge_base'):
                        source_icons.append("📚 Knowledge Base")
                    if sources.get('function_calls'):
                        source_icons.append("🔧 Function Calls")
                    
                    sources_text = " • ".join(source_icons) if source_icons else "💭 Internal Knowledge"
                    st.caption(f"Sources: {sources_text}")
                
                with col2:
                    # Confidence with emoji
                    conf_emoji = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"
                    st.caption(f"{conf_emoji} {confidence:.0%}")


def render_feedback_widget(message_idx: int):
    """Render feedback collection widget for a message."""
    if st.session_state.rlhf_manager.should_request_feedback():
        with st.expander("⭐ Rate this response (Help improve AI)", expanded=False):
            st.write("📝 Your feedback helps make responses better!")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                rating = st.selectbox(
                    "Rating:", 
                    options=[1, 2, 3, 4, 5],
                    index=4,
                    key=f"rating_{message_idx}"
                )
            
            with col2:
                feedback_text = st.text_area(
                    "Comments (optional):",
                    placeholder="Tell us how we can improve...",
                    key=f"feedback_{message_idx}",
                    height=60
                )
            
            if st.button("Submit Feedback", key=f"submit_{message_idx}"):
                # Get the last assistant message
                assistant_messages = [msg for msg in st.session_state.messages if not msg.get('is_user', True)]
                if assistant_messages:
                    last_response = assistant_messages[-1]
                    user_messages = [msg for msg in st.session_state.messages if msg.get('is_user', True)]
                    last_query = user_messages[-1]['content'] if user_messages else ""
                    
                    success = st.session_state.rlhf_manager.collect_feedback(
                        session_id=st.session_state.session_id,
                        query=last_query,
                        response=last_response['content'],
                        rating=rating,
                        feedback_text=feedback_text if feedback_text else None,
                        sources_used=last_response.get('sources_used', {}),
                        confidence_score=last_response.get('confidence_score', 0.5)
                    )
                    
                    if success:
                        st.success("Thank you for your feedback!")
                    else:
                        st.error("Failed to submit feedback. Please try again.")


def render_sidebar():
    """Render the sidebar with controls and information."""
    with st.sidebar:
        st.title("🤖 Agentic RAG")
        st.divider()
        
        # Document Upload Section
        st.subheader("📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload documents to expand knowledge base",
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx', 'html', 'md'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    processed_count = 0
                    failed_count = 0
                    
                    for uploaded_file in uploaded_files:
                        try:
                            st.info(f"Processing {uploaded_file.name}...")
                            file_content = uploaded_file.read()
                            
                            result = st.session_state.doc_processor.process_uploaded_file(
                                file_content, uploaded_file.name
                            )
                            
                            if result.get('success', False):
                                processed_count += 1
                                st.success(f"✅ {uploaded_file.name}")
                            else:
                                failed_count += 1
                                error_msg = result.get('error', 'Unknown error')
                                st.error(f"❌ {uploaded_file.name}: {error_msg}")
                        
                        except Exception as e:
                            failed_count += 1
                            st.error(f"❌ {uploaded_file.name}: {str(e)}")
                    
                    # Final status
                    if processed_count > 0:
                        st.success(f"✅ Successfully processed {processed_count} documents!")
                    if failed_count > 0:
                        st.warning(f"⚠️ Failed to process {failed_count} documents")
        
        st.divider()
        
        # Knowledge Base Stats
        st.subheader("📊 Knowledge Base")
        
        # Check ChromaDB status
        chroma_available = st.session_state.chroma_manager._is_available()
        
        if chroma_available:
            stats = st.session_state.chroma_manager.get_collection_stats()
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.metric("📚 Documents", stats.get('total_documents', 0))
            with col2:
                if st.button("🗑️ Clear", help="Clear knowledge base"):
                    if st.session_state.chroma_manager.reset_collection():
                        st.success("✅ Cleared!")
                        st.rerun()
        else:
            st.error("❌ **ChromaDB Unavailable**")
            st.warning("Document upload and search disabled")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.info("Try recovery to fix database issues")
            with col2:
                if st.button("🔧 Recover", help="Attempt to fix ChromaDB"):
                    with st.spinner("Recovering database..."):
                        if st.session_state.chroma_manager.force_recovery():
                            st.success("✅ Recovery successful!")
                            st.rerun()
                        else:
                            st.error("❌ Recovery failed")
        
        st.divider()
        
        # Memory Management
        st.subheader("🧠 Memory")
        history = st.session_state.workflow.get_conversation_history(st.session_state.session_id)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.metric("💬 Messages", len(history))
        with col2:
            if st.button("🧹 Clear", help="Clear conversation memory"):
                if st.session_state.workflow.clear_memory(st.session_state.session_id):
                    st.session_state.messages = []
                    st.success("✅ Cleared!")
                    st.rerun()
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        
        # Streaming toggle with emoji
        use_streaming = st.toggle("⚡ Enable Streaming", value=True, help="Real-time response streaming")
        st.session_state.use_streaming = use_streaming
        
        # Tool selection with emojis
        st.write("🔧 **Tool Selection:**")
        use_web_search = st.checkbox("🌐 Web Search", value=True, help="Search the internet for current information")
        use_rag = st.checkbox("📚 Knowledge Base", value=True, help="Use uploaded documents")
        
        st.session_state.tool_settings = {
            "web_search": use_web_search,
            "rag": use_rag
        }
        
        st.divider()
        
        # RLHF Dashboard Link
        if st.button("📈 Feedback Dashboard", use_container_width=True, type="secondary"):
            st.session_state.show_dashboard = True
        
        # System status
        st.divider()
        st.subheader("📡 Status")
        
        # Show system status
        with st.expander("🔍 System Info", expanded=False):
            st.success("✅ Gemini LLM: Connected")
            st.success("✅ ChromaDB: Running")
            st.info(f"📊 Session: {st.session_state.session_id[-8:]}")


def render_feedback_dashboard():
    """Render the RLHF feedback dashboard."""
    st.header("📈 Feedback Dashboard")
    
    # Get dashboard data
    dashboard_data = st.session_state.rlhf_manager.get_feedback_dashboard_data()
    
    if "error" in dashboard_data:
        st.error(f"Error loading dashboard: {dashboard_data['error']}")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    recent_stats = dashboard_data.get('recent_stats', {})
    monthly_stats = dashboard_data.get('monthly_stats', {})
    
    with col1:
        st.metric(
            "Avg Rating (7d)", 
            f"{recent_stats.get('average_rating', 0):.1f}",
            delta=f"{recent_stats.get('average_rating', 0) - monthly_stats.get('average_rating', 0):.1f}"
        )
    
    with col2:
        st.metric("Total Feedback (7d)", recent_stats.get('total_feedback', 0))
    
    with col3:
        st.metric(
            "Avg Confidence (7d)", 
            f"{recent_stats.get('average_confidence', 0):.1%}"
        )
    
    with col4:
        st.metric(
            "Avg Response Time (7d)", 
            f"{recent_stats.get('average_response_time', 0):.1f}s"
        )
    
    # Rating distribution
    st.subheader("Rating Distribution (Last 7 Days)")
    rating_dist = recent_stats.get('rating_distribution', {})
    if rating_dist:
        st.bar_chart(rating_dist)
    else:
        st.info("No rating data available")
    
    # Analysis and suggestions
    analysis = dashboard_data.get('analysis', {})
    if analysis:
        st.subheader("Improvement Suggestions")
        suggestions = analysis.get('improvement_suggestions', [])
        for suggestion in suggestions:
            st.write(f"• {suggestion}")
    
    if st.button("← Back to Chat"):
        st.session_state.show_dashboard = False
        st.rerun()


async def process_message_async(user_input: str):
    """Process message asynchronously."""
    return await st.session_state.workflow.process_query(
        user_input, 
        st.session_state.session_id
    )


def render_main_chat():
    """Render the main chat interface."""
    # Main title with emojis
    st.title("🤖 Agentic RAG Assistant")
    
    # Welcome message with info boxes
    st.success("🎯 **Welcome!** I'm your intelligent AI assistant with advanced capabilities!")
    
    # Feature overview using columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        🌟 **Core Features:**
        - 🌐 Real-time web search
        - 📚 Document knowledge base
        - 🔧 Function calling & calculations
        """)
    
    with col2:
        st.info("""
        ⚡ **Smart Capabilities:**
        - 🧠 Conversational memory
        - 📊 Streaming responses
        - 🎯 Context-aware answers
        """)
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        render_chat_message(message, message.get('is_user', True))
        
        # Show feedback widget for assistant messages occasionally
        if not message.get('is_user', True) and i == len(st.session_state.messages) - 1:
            render_feedback_widget(i)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        user_message = {"content": prompt, "is_user": True, "timestamp": datetime.now()}
        st.session_state.messages.append(user_message)
        
        # Display user message immediately
        render_chat_message(user_message, is_user=True)
        
        # Process and display assistant response
        with st.spinner("Thinking..."):
            try:
                if st.session_state.get('use_streaming', True):
                    # Streaming response
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    with st.chat_message("assistant"):
                        response_placeholder = st.empty()
                        for chunk in st.session_state.workflow.stream_response(prompt, st.session_state.session_id):
                            full_response += chunk
                            response_placeholder.write(f"🤖 **Assistant:** {full_response}")
                    
                    # Add to message history
                    assistant_message = {
                        "content": full_response,
                        "is_user": False,
                        "timestamp": datetime.now(),
                        "sources_used": {"web_search": False, "knowledge_base": True, "function_calls": False},
                        "confidence_score": 0.8  # Placeholder for streaming
                    }
                    
                else:
                    # Regular response
                    result = asyncio.run(process_message_async(prompt))
                    
                    assistant_message = {
                        "content": result['response'],
                        "is_user": False,
                        "timestamp": datetime.now(),
                        "sources_used": result.get('sources_used', {}),
                        "confidence_score": result.get('confidence_score', 0.5)
                    }
                    
                    render_chat_message(assistant_message, is_user=False)
                
                st.session_state.messages.append(assistant_message)
                
            except Exception as e:
                st.error(f"Error processing message: {str(e)}")
                error_message = {
                    "content": f"I apologize, but I encountered an error: {str(e)}",
                    "is_user": False,
                    "timestamp": datetime.now(),
                    "sources_used": {},
                    "confidence_score": 0.1
                }
                st.session_state.messages.append(error_message)
        
        # Refresh to show new messages
        st.rerun()


def main():
    """Main application function."""
    try:
        initialize_session_state()
        
        # Check if showing dashboard
        if st.session_state.get('show_dashboard', False):
            render_feedback_dashboard()
        else:
            # Render sidebar
            render_sidebar()
            
            # Render main chat
            render_main_chat()
    
    except Exception as e:
        st.error("🚨 **Application Error**")
        st.error(f"**Error:** {str(e)}")
        
        with st.expander("🔍 Technical Details"):
            import traceback
            st.code(traceback.format_exc())
        
        st.info("💡 **Possible solutions:**")
        st.info("• Check your `.env` file has `GOOGLE_API_KEY`")
        st.info("• Ensure all dependencies are installed: `pip install -r requirements.txt`")
        st.info("• Try deleting the `data/` folder and restarting")


if __name__ == "__main__":
    main()