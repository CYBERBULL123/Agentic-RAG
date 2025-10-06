"""
Persistent Memory Management for Agentic RAG System
Handles conversation history, user preferences, and contextual knowledge
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import sqlite3
from loguru import logger


@dataclass 
class ConversationMessage:
    """Individual message in a conversation."""
    message_id: str
    session_id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UserProfile:
    """User profile with preferences and context."""
    user_id: str
    preferences: Dict[str, Any]
    interests: List[str]
    expertise_areas: List[str]
    conversation_style: str
    last_updated: str
    total_interactions: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryManager:
    """Manages persistent memory for the agentic system."""
    
    def __init__(self, 
                 memory_db_path: str = "./data/memory.db",
                 max_session_messages: int = 50,
                 memory_retention_days: int = 30):
        """
        Initialize memory manager.
        
        Args:
            memory_db_path: Path to SQLite memory database
            max_session_messages: Maximum messages per session
            memory_retention_days: Days to retain memory
        """
        self.memory_db_path = memory_db_path
        self.max_session_messages = max_session_messages
        self.memory_retention_days = memory_retention_days
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(memory_db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # In-memory cache for active sessions
        self.session_cache = {}
        
        logger.info(f"Initialized MemoryManager with DB: {memory_db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.cursor()
                
                # Conversation messages table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        message_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        metadata TEXT,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    )
                """)
                
                # Sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        created_at TEXT NOT NULL,
                        last_activity TEXT NOT NULL,
                        message_count INTEGER DEFAULT 0,
                        session_metadata TEXT
                    )
                """)
                
                # User profiles table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id TEXT PRIMARY KEY,
                        preferences TEXT NOT NULL,
                        interests TEXT NOT NULL,
                        expertise_areas TEXT NOT NULL,
                        conversation_style TEXT,
                        last_updated TEXT NOT NULL,
                        total_interactions INTEGER DEFAULT 0
                    )
                """)
                
                # Context knowledge table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS context_knowledge (
                        context_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        session_id TEXT,
                        knowledge_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        relevance_score REAL DEFAULT 1.0,
                        created_at TEXT NOT NULL,
                        last_accessed TEXT NOT NULL,
                        access_count INTEGER DEFAULT 1
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_user ON context_knowledge(user_id)")
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def add_message(self, 
                   session_id: str,
                   role: str,
                   content: str,
                   metadata: Optional[Dict[str, Any]] = None,
                   user_id: Optional[str] = None) -> bool:
        """
        Add a message to the conversation history.
        
        Args:
            session_id: Session identifier
            role: Message role (user/assistant)
            content: Message content
            metadata: Optional message metadata
            user_id: Optional user identifier
            
        Returns:
            Success status
        """
        try:
            message_id = f"{session_id}_{datetime.now().isoformat()}_{role}"
            timestamp = datetime.now().isoformat()
            
            # Create session if it doesn't exist
            self._ensure_session_exists(session_id, user_id)
            
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.cursor()
                
                # Insert message
                cursor.execute("""
                    INSERT INTO messages 
                    (message_id, session_id, role, content, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    message_id, 
                    session_id, 
                    role, 
                    content, 
                    timestamp,
                    json.dumps(metadata) if metadata else None
                ))
                
                # Update session info
                cursor.execute("""
                    UPDATE sessions 
                    SET last_activity = ?, message_count = message_count + 1
                    WHERE session_id = ?
                """, (timestamp, session_id))
                
                conn.commit()
                
                # Update cache
                if session_id in self.session_cache:
                    self.session_cache[session_id].append({
                        "message_id": message_id,
                        "role": role,
                        "content": content,
                        "timestamp": timestamp,
                        "metadata": metadata
                    })
                    
                    # Maintain cache size
                    if len(self.session_cache[session_id]) > self.max_session_messages:
                        self.session_cache[session_id].pop(0)
                
                logger.info(f"Added message to session {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            return False
    
    def get_session_history(self, 
                           session_id: str,
                           limit: Optional[int] = None,
                           include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Optional limit on number of messages
            include_metadata: Whether to include message metadata
            
        Returns:
            List of messages
        """
        try:
            # Check cache first
            if session_id in self.session_cache:
                cached_messages = self.session_cache[session_id]
                if limit:
                    cached_messages = cached_messages[-limit:]
                return cached_messages
            
            # Query database
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT message_id, role, content, timestamp, metadata
                    FROM messages 
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query, (session_id,))
                rows = cursor.fetchall()
                
                messages = []
                for row in rows:
                    message = {
                        "message_id": row[0],
                        "role": row[1],
                        "content": row[2],
                        "timestamp": row[3]
                    }
                    
                    if include_metadata and row[4]:
                        try:
                            message["metadata"] = json.loads(row[4])
                        except:
                            message["metadata"] = {}
                    
                    messages.append(message)
                
                # Reverse to get chronological order
                messages.reverse()
                
                # Update cache
                self.session_cache[session_id] = messages
                
                return messages
                
        except Exception as e:
            logger.error(f"Error getting session history: {e}")
            return []
    
    def get_user_context(self, user_id: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get user context from recent conversations and knowledge.
        
        Args:
            user_id: User identifier
            limit: Number of recent sessions to include
            
        Returns:
            User context information
        """
        try:
            context = {
                "user_id": user_id,
                "profile": self.get_user_profile(user_id),
                "recent_sessions": [],
                "knowledge_items": [],
                "conversation_patterns": {}
            }
            
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.cursor()
                
                # Get recent sessions
                cursor.execute("""
                    SELECT session_id, created_at, last_activity, message_count
                    FROM sessions 
                    WHERE user_id = ?
                    ORDER BY last_activity DESC
                    LIMIT ?
                """, (user_id, limit))
                
                sessions = cursor.fetchall()
                for session in sessions:
                    context["recent_sessions"].append({
                        "session_id": session[0],
                        "created_at": session[1],
                        "last_activity": session[2],
                        "message_count": session[3]
                    })
                
                # Get relevant knowledge items
                cursor.execute("""
                    SELECT content, knowledge_type, relevance_score, created_at
                    FROM context_knowledge
                    WHERE user_id = ?
                    ORDER BY relevance_score DESC, last_accessed DESC
                    LIMIT ?
                """, (user_id, limit))
                
                knowledge = cursor.fetchall()
                for item in knowledge:
                    context["knowledge_items"].append({
                        "content": item[0],
                        "type": item[1],
                        "relevance": item[2],
                        "created_at": item[3]
                    })
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return {"user_id": user_id, "error": str(e)}
    
    def add_contextual_knowledge(self, 
                               user_id: str,
                               session_id: str,
                               knowledge_type: str,
                               content: str,
                               relevance_score: float = 1.0) -> bool:
        """
        Add contextual knowledge about the user.
        
        Args:
            user_id: User identifier
            session_id: Current session
            knowledge_type: Type of knowledge (preference, fact, etc.)
            content: Knowledge content
            relevance_score: Relevance score (0-1)
            
        Returns:
            Success status
        """
        try:
            context_id = f"{user_id}_{knowledge_type}_{datetime.now().isoformat()}"
            timestamp = datetime.now().isoformat()
            
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO context_knowledge
                    (context_id, user_id, session_id, knowledge_type, content, 
                     relevance_score, created_at, last_accessed, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    context_id, user_id, session_id, knowledge_type, content,
                    relevance_score, timestamp, timestamp, 1
                ))
                
                conn.commit()
                logger.info(f"Added contextual knowledge for user {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error adding contextual knowledge: {e}")
            return False
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile."""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT preferences, interests, expertise_areas, conversation_style,
                           last_updated, total_interactions
                    FROM user_profiles
                    WHERE user_id = ?
                """, (user_id,))
                
                row = cursor.fetchone()
                if row:
                    return UserProfile(
                        user_id=user_id,
                        preferences=json.loads(row[0]),
                        interests=json.loads(row[1]),
                        expertise_areas=json.loads(row[2]),
                        conversation_style=row[3],
                        last_updated=row[4],
                        total_interactions=row[5]
                    )
                return None
                
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None
    
    def update_user_profile(self, profile: UserProfile) -> bool:
        """Update or create user profile."""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO user_profiles
                    (user_id, preferences, interests, expertise_areas, 
                     conversation_style, last_updated, total_interactions)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    profile.user_id,
                    json.dumps(profile.preferences),
                    json.dumps(profile.interests),
                    json.dumps(profile.expertise_areas),
                    profile.conversation_style,
                    profile.last_updated,
                    profile.total_interactions
                ))
                
                conn.commit()
                logger.info(f"Updated profile for user {profile.user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            return False
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a specific session."""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.cursor()
                
                # Delete messages
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                
                # Delete session
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                
                conn.commit()
                
                # Clear from cache
                if session_id in self.session_cache:
                    del self.session_cache[session_id]
                
                logger.info(f"Cleared session {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return False
    
    def cleanup_old_data(self) -> Dict[str, int]:
        """Clean up old data based on retention policy."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=self.memory_retention_days)).isoformat()
            
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.cursor()
                
                # Count old data
                cursor.execute("SELECT COUNT(*) FROM messages WHERE timestamp < ?", (cutoff_date,))
                old_messages = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM sessions WHERE last_activity < ?", (cutoff_date,))
                old_sessions = cursor.fetchone()[0]
                
                # Delete old data
                cursor.execute("DELETE FROM messages WHERE timestamp < ?", (cutoff_date,))
                cursor.execute("DELETE FROM sessions WHERE last_activity < ?", (cutoff_date,))
                cursor.execute("DELETE FROM context_knowledge WHERE created_at < ?", (cutoff_date,))
                
                conn.commit()
                
                # Clear affected cache entries
                self.session_cache.clear()
                
                logger.info(f"Cleaned up {old_messages} messages and {old_sessions} sessions")
                return {"messages": old_messages, "sessions": old_sessions}
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return {"error": str(e)}
    
    def _ensure_session_exists(self, session_id: str, user_id: Optional[str] = None):
        """Ensure session exists in database."""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.cursor()
                
                # Check if session exists
                cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
                
                if not cursor.fetchone():
                    # Create session
                    timestamp = datetime.now().isoformat()
                    cursor.execute("""
                        INSERT INTO sessions 
                        (session_id, user_id, created_at, last_activity, message_count)
                        VALUES (?, ?, ?, ?, ?)
                    """, (session_id, user_id, timestamp, timestamp, 0))
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error ensuring session exists: {e}")


# Global instance
_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        from config.settings import config
        _memory_manager = MemoryManager(
            max_session_messages=config.max_memory_messages
        )
    return _memory_manager