"""
RLHF (Reinforcement Learning from Human Feedback) Manager
Handles feedback collection, analysis, and system improvement
"""

import json
import os
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import statistics
from loguru import logger
import random


@dataclass
class Feedback:
    """Individual feedback record."""
    feedback_id: str
    session_id: str
    user_query: str
    assistant_response: str
    rating: int  # 1-5 scale
    feedback_text: Optional[str]
    sources_used: Dict[str, bool]
    confidence_score: float
    response_time: float
    timestamp: str
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeedbackAnalysis:
    """Analysis of feedback data."""
    total_feedback: int
    average_rating: float
    rating_distribution: Dict[int, int]
    average_confidence: float
    average_response_time: float
    improvement_suggestions: List[str]
    source_effectiveness: Dict[str, float]
    time_period: str


class RLHFManager:
    """Manages RLHF feedback collection and analysis."""
    
    def __init__(self,
                 feedback_db_path: str = "./data/feedback.db",
                 collection_rate: float = 0.3,
                 min_rating_for_positive: int = 4,
                 enable_rlhf: bool = True):
        """
        Initialize RLHF manager.
        
        Args:
            feedback_db_path: Path to feedback database
            collection_rate: Rate at which to request feedback (0-1)
            min_rating_for_positive: Minimum rating considered positive
            enable_rlhf: Whether RLHF is enabled
        """
        self.feedback_db_path = feedback_db_path
        self.collection_rate = collection_rate
        self.min_rating_for_positive = min_rating_for_positive
        self.enable_rlhf = enable_rlhf
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(feedback_db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Initialized RLHFManager with collection rate: {collection_rate}")
    
    def _init_database(self):
        """Initialize SQLite database for feedback storage."""
        try:
            with sqlite3.connect(self.feedback_db_path) as conn:
                cursor = conn.cursor()
                
                # Feedback table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        feedback_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        user_query TEXT NOT NULL,
                        assistant_response TEXT NOT NULL,
                        rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                        feedback_text TEXT,
                        sources_used TEXT NOT NULL,
                        confidence_score REAL NOT NULL,
                        response_time REAL DEFAULT 0.0,
                        timestamp TEXT NOT NULL,
                        user_id TEXT
                    )
                """)
                
                # Analytics summary table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analytics_summary (
                        summary_id TEXT PRIMARY KEY,
                        time_period TEXT NOT NULL,
                        total_feedback INTEGER DEFAULT 0,
                        average_rating REAL DEFAULT 0.0,
                        rating_distribution TEXT,
                        average_confidence REAL DEFAULT 0.0,
                        average_response_time REAL DEFAULT 0.0,
                        source_effectiveness TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
                
                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id)")
                
                conn.commit()
                logger.info("Feedback database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing feedback database: {e}")
    
    def should_request_feedback(self) -> bool:
        """
        Determine if feedback should be requested based on collection rate.
        
        Returns:
            True if feedback should be requested
        """
        if not self.enable_rlhf:
            return False
        
        return random.random() < self.collection_rate
    
    def collect_feedback(self,
                        session_id: str,
                        query: str,
                        response: str,
                        rating: int,
                        sources_used: Dict[str, bool],
                        confidence_score: float,
                        feedback_text: Optional[str] = None,
                        response_time: float = 0.0,
                        user_id: Optional[str] = None) -> bool:
        """
        Collect and store user feedback.
        
        Args:
            session_id: Session identifier
            query: User query
            response: Assistant response
            rating: User rating (1-5)
            sources_used: Sources used in response
            confidence_score: Model confidence score
            feedback_text: Optional feedback text
            response_time: Response generation time
            user_id: Optional user identifier
            
        Returns:
            Success status
        """
        try:
            if not self.enable_rlhf:
                return True
            
            # Validate rating
            if not (1 <= rating <= 5):
                logger.error(f"Invalid rating: {rating}. Must be 1-5")
                return False
            
            feedback_id = f"fb_{session_id}_{datetime.now().isoformat()}"
            timestamp = datetime.now().isoformat()
            
            feedback = Feedback(
                feedback_id=feedback_id,
                session_id=session_id,
                user_query=query[:1000],  # Truncate long queries
                assistant_response=response[:2000],  # Truncate long responses
                rating=rating,
                feedback_text=feedback_text[:500] if feedback_text else None,  # Truncate feedback
                sources_used=sources_used,
                confidence_score=confidence_score,
                response_time=response_time,
                timestamp=timestamp,
                user_id=user_id
            )
            
            with sqlite3.connect(self.feedback_db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO feedback
                    (feedback_id, session_id, user_query, assistant_response, rating,
                     feedback_text, sources_used, confidence_score, response_time, 
                     timestamp, user_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.feedback_id,
                    feedback.session_id,
                    feedback.user_query,
                    feedback.assistant_response,
                    feedback.rating,
                    feedback.feedback_text,
                    json.dumps(feedback.sources_used),
                    feedback.confidence_score,
                    feedback.response_time,
                    feedback.timestamp,
                    feedback.user_id
                ))
                
                conn.commit()
                
            logger.info(f"Collected feedback: ID {feedback_id}, Rating {rating}")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            return False
    
    def get_feedback_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Get feedback statistics for the specified period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Statistics dictionary
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.feedback_db_path) as conn:
                cursor = conn.cursor()
                
                # Get all feedback from the period
                cursor.execute("""
                    SELECT rating, confidence_score, response_time, sources_used
                    FROM feedback 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """, (cutoff_date,))
                
                rows = cursor.fetchall()
                
                if not rows:
                    return {
                        "total_feedback": 0,
                        "average_rating": 0,
                        "rating_distribution": {},
                        "average_confidence": 0,
                        "average_response_time": 0,
                        "source_effectiveness": {}
                    }
                
                # Calculate statistics
                ratings = [row[0] for row in rows]
                confidences = [row[1] for row in rows]
                response_times = [row[2] for row in rows]
                
                # Rating distribution
                rating_dist = {i: ratings.count(i) for i in range(1, 6)}
                
                # Source effectiveness analysis
                source_effectiveness = {}
                source_ratings = {"web_search": [], "knowledge_base": [], "function_calls": []}
                
                for row in rows:
                    try:
                        sources = json.loads(row[3])
                        rating = row[0]
                        
                        for source, used in sources.items():
                            if used and source in source_ratings:
                                source_ratings[source].append(rating)
                    except:
                        continue
                
                for source, ratings_list in source_ratings.items():
                    if ratings_list:
                        source_effectiveness[source] = statistics.mean(ratings_list)
                    else:
                        source_effectiveness[source] = 0
                
                stats = {
                    "total_feedback": len(ratings),
                    "average_rating": statistics.mean(ratings),
                    "rating_distribution": rating_dist,
                    "average_confidence": statistics.mean(confidences),
                    "average_response_time": statistics.mean(response_times),
                    "source_effectiveness": source_effectiveness
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting feedback stats: {e}")
            return {"error": str(e)}
    
    def analyze_feedback_trends(self, days: int = 30) -> FeedbackAnalysis:
        """
        Analyze feedback trends and generate improvement suggestions.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Feedback analysis object
        """
        try:
            stats = self.get_feedback_stats(days)
            
            if stats.get("total_feedback", 0) == 0:
                return FeedbackAnalysis(
                    total_feedback=0,
                    average_rating=0,
                    rating_distribution={},
                    average_confidence=0,
                    average_response_time=0,
                    improvement_suggestions=["No feedback available for analysis"],
                    source_effectiveness={},
                    time_period=f"Last {days} days"
                )
            
            # Generate improvement suggestions
            suggestions = []
            
            avg_rating = stats["average_rating"]
            if avg_rating < 3.5:
                suggestions.append("Overall satisfaction is low. Review response quality and relevance.")
            
            if avg_rating < 4.0:
                suggestions.append("Consider improving response accuracy and helpfulness.")
            
            # Analyze source effectiveness
            source_eff = stats["source_effectiveness"]
            if source_eff.get("web_search", 0) < 3.5:
                suggestions.append("Web search results may need better filtering and summarization.")
            
            if source_eff.get("knowledge_base", 0) < 3.5:
                suggestions.append("Knowledge base retrieval could be improved with better document processing.")
            
            # Response time analysis
            avg_time = stats["average_response_time"]
            if avg_time > 5.0:
                suggestions.append("Response times are slow. Consider optimizing model inference.")
            
            # Confidence analysis
            avg_conf = stats["average_confidence"]
            if avg_conf < 0.7:
                suggestions.append("Model confidence is low. Review training data and model parameters.")
            
            # Rating distribution analysis
            rating_dist = stats["rating_distribution"]
            low_ratings = rating_dist.get(1, 0) + rating_dist.get(2, 0)
            total_ratings = stats["total_feedback"]
            
            if low_ratings / total_ratings > 0.2:  # More than 20% low ratings
                suggestions.append("High percentage of low ratings. Focus on understanding user needs better.")
            
            if not suggestions:
                suggestions.append("Performance looks good! Keep up the excellent work.")
            
            return FeedbackAnalysis(
                total_feedback=stats["total_feedback"],
                average_rating=avg_rating,
                rating_distribution=stats["rating_distribution"],
                average_confidence=avg_conf,
                average_response_time=avg_time,
                improvement_suggestions=suggestions,
                source_effectiveness=source_eff,
                time_period=f"Last {days} days"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing feedback trends: {e}")
            return FeedbackAnalysis(
                total_feedback=0,
                average_rating=0,
                rating_distribution={},
                average_confidence=0,
                average_response_time=0,
                improvement_suggestions=[f"Error in analysis: {str(e)}"],
                source_effectiveness={},
                time_period=f"Last {days} days"
            )
    
    def get_feedback_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive dashboard data for feedback analysis.
        
        Returns:
            Dashboard data with multiple time periods
        """
        try:
            # Get data for different time periods
            recent_stats = self.get_feedback_stats(7)  # Last 7 days
            monthly_stats = self.get_feedback_stats(30)  # Last 30 days
            
            # Get analysis
            analysis = self.analyze_feedback_trends(30)
            
            # Get recent feedback for examples
            with sqlite3.connect(self.feedback_db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT rating, feedback_text, timestamp
                    FROM feedback 
                    ORDER BY timestamp DESC
                    LIMIT 10
                """)
                
                recent_feedback = []
                for row in cursor.fetchall():
                    recent_feedback.append({
                        "rating": row[0],
                        "text": row[1],
                        "timestamp": row[2]
                    })
            
            return {
                "recent_stats": recent_stats,
                "monthly_stats": monthly_stats,
                "analysis": analysis.to_dict(),
                "recent_feedback": recent_feedback,
                "collection_enabled": self.enable_rlhf,
                "collection_rate": self.collection_rate
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {"error": str(e)}
    
    def export_feedback(self, days: int = 30, format: str = "json") -> Dict[str, Any]:
        """
        Export feedback data for analysis.
        
        Args:
            days: Number of days to export
            format: Export format (json, csv)
            
        Returns:
            Export result
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.feedback_db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM feedback 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """, (cutoff_date,))
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                if format == "json":
                    data = []
                    for row in rows:
                        data.append(dict(zip(columns, row)))
                    
                    return {
                        "success": True,
                        "data": data,
                        "format": "json",
                        "records": len(data)
                    }
                
                else:
                    return {
                        "success": False,
                        "error": f"Unsupported format: {format}"
                    }
            
        except Exception as e:
            logger.error(f"Error exporting feedback: {e}")
            return {"success": False, "error": str(e)}
    
    def clear_old_feedback(self, days: int = 90) -> Dict[str, int]:
        """
        Clear feedback older than specified days.
        
        Args:
            days: Age threshold for deletion
            
        Returns:
            Deletion statistics
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.feedback_db_path) as conn:
                cursor = conn.cursor()
                
                # Count old feedback
                cursor.execute("SELECT COUNT(*) FROM feedback WHERE timestamp < ?", (cutoff_date,))
                old_count = cursor.fetchone()[0]
                
                # Delete old feedback
                cursor.execute("DELETE FROM feedback WHERE timestamp < ?", (cutoff_date,))
                conn.commit()
                
                logger.info(f"Cleared {old_count} old feedback records")
                return {"deleted": old_count}
                
        except Exception as e:
            logger.error(f"Error clearing old feedback: {e}")
            return {"error": str(e)}


# Global instance
_rlhf_manager = None

def get_rlhf_manager() -> RLHFManager:
    """Get global RLHF manager instance."""
    global _rlhf_manager
    if _rlhf_manager is None:
        from config.settings import config
        _rlhf_manager = RLHFManager(
            collection_rate=config.feedback_collection_rate,
            enable_rlhf=config.enable_rlhf
        )
    return _rlhf_manager