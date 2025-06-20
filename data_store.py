
"""
Data store module for AI-powered social campaign optimizer.
Handles audit logging, data persistence, and caching.
"""
import logging
import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
from pathlib import Path
from contextlib import contextmanager
from config import config

logger = logging.getLogger(__name__)

class DataStore:
    """
    Central data store for campaign optimization data.
    Handles audit logging, performance data caching, and insights storage.
    """
    
    def __init__(self, db_path: str = "campaign_optimizer.db"):
        self.db_path = db_path
        self.init_database()
        logger.info(f"DataStore initialized with database: {db_path}")
    
    def init_database(self):
        """Initialize SQLite database with required tables."""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Audit log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    action VARCHAR(100) NOT NULL,
                    details TEXT,
                    user_id VARCHAR(50),
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT
                )
            """)
            
            # Performance data cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id VARCHAR(50),
                    campaign_id VARCHAR(50),
                    adset_id VARCHAR(50),
                    ad_id VARCHAR(50),
                    date_start DATE,
                    date_end DATE,
                    metric_name VARCHAR(100),
                    metric_value REAL,
                    cached_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME
                )
            """)
            
            # AI insights storage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    insight_type VARCHAR(50),
                    entity_id VARCHAR(100),
                    insight_data TEXT,
                    confidence_score REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    applied BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Anomaly detection results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name VARCHAR(100),
                    entity_id VARCHAR(100),
                    anomaly_type VARCHAR(50),
                    severity VARCHAR(20),
                    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT FALSE,
                    explanation TEXT
                )
            """)
            
            # Automation history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS automation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_type VARCHAR(100),
                    entity_id VARCHAR(100),
                    parameters TEXT,
                    executed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    dry_run BOOLEAN DEFAULT TRUE,
                    success BOOLEAN,
                    result_data TEXT
                )
            """)
            
            conn.commit()
            logger.info("Database tables initialized successfully")
    
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def log_audit_entry(self, action: str, details: Dict[str, Any] = None, 
                       user_id: str = None, success: bool = True, 
                       error_message: str = None):
        """
        Log an audit entry for tracking all system actions.
        
        Args:
            action: Action performed
            details: Additional details as dictionary
            user_id: User who performed the action
            success: Whether the action was successful
            error_message: Error message if action failed
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO audit_log (action, details, user_id, success, error_message)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    action,
                    json.dumps(details) if details else None,
                    user_id,
                    success,
                    error_message
                ))
                conn.commit()
                logger.debug(f"Audit entry logged: {action}")
        except Exception as e:
            logger.error(f"Failed to log audit entry: {e}")
    
    def cache_performance_data(self, performance_data: List[Dict[str, Any]], 
                              cache_duration_hours: int = 24):
        """
        Cache performance data to reduce API calls.
        
        Args:
            performance_data: List of performance metrics
            cache_duration_hours: How long to cache the data
        """
        try:
            expires_at = datetime.now() + timedelta(hours=cache_duration_hours)
            
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                for data in performance_data:
                    cursor.execute("""
                        INSERT OR REPLACE INTO performance_cache 
                        (account_id, campaign_id, adset_id, ad_id, date_start, 
                         date_end, metric_name, metric_value, expires_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        data.get('account_id'),
                        data.get('campaign_id'),
                        data.get('adset_id'),
                        data.get('ad_id'),
                        data.get('date_start'),
                        data.get('date_end'),
                        data.get('metric_name'),
                        data.get('metric_value'),
                        expires_at
                    ))
                
                conn.commit()
                logger.info(f"Cached {len(performance_data)} performance metrics")
                
        except Exception as e:
            logger.error(f"Failed to cache performance data: {e}")
    
    def get_cached_performance_data(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve cached performance data.
        
        Args:
            filters: Filter criteria for the data
            
        Returns:
            List of cached performance metrics
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Base query
                query = """
                    SELECT * FROM performance_cache 
                    WHERE expires_at > datetime('now')
                """
                params = []
                
                # Add filters
                if filters:
                    if filters.get('account_id'):
                        query += " AND account_id = ?"
                        params.append(filters['account_id'])
                    if filters.get('campaign_id'):
                        query += " AND campaign_id = ?"
                        params.append(filters['campaign_id'])
                    if filters.get('date_start'):
                        query += " AND date_start >= ?"
                        params.append(filters['date_start'])
                    if filters.get('date_end'):
                        query += " AND date_end <= ?"
                        params.append(filters['date_end'])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to retrieve cached data: {e}")
            return []
    
    def store_ai_insight(self, insight_type: str, entity_id: str, 
                        insight_data: Dict[str, Any], confidence_score: float = 0.0):
        """
        Store AI-generated insights.
        
        Args:
            insight_type: Type of insight (e.g., 'optimization', 'anomaly_explanation')
            entity_id: ID of the entity the insight relates to
            insight_data: The insight data
            confidence_score: Confidence score of the insight
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO ai_insights (insight_type, entity_id, insight_data, confidence_score)
                    VALUES (?, ?, ?, ?)
                """, (
                    insight_type,
                    entity_id,
                    json.dumps(insight_data),
                    confidence_score
                ))
                conn.commit()
                logger.info(f"AI insight stored: {insight_type} for {entity_id}")
                
        except Exception as e:
            logger.error(f"Failed to store AI insight: {e}")
    
    def get_ai_insights(self, insight_type: str = None, entity_id: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve AI insights.
        
        Args:
            insight_type: Filter by insight type
            entity_id: Filter by entity ID
            
        Returns:
            List of AI insights
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM ai_insights WHERE 1=1"
                params = []
                
                if insight_type:
                    query += " AND insight_type = ?"
                    params.append(insight_type)
                if entity_id:
                    query += " AND entity_id = ?"
                    params.append(entity_id)
                
                query += " ORDER BY created_at DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                insights = []
                for row in rows:
                    insight = dict(row)
                    insight['insight_data'] = json.loads(insight['insight_data'])
                    insights.append(insight)
                
                return insights
                
        except Exception as e:
            logger.error(f"Failed to retrieve AI insights: {e}")
            return []
    
    def record_anomaly(self, metric_name: str, entity_id: str, 
                      anomaly_type: str, severity: str, explanation: str = None):
        """
        Record an anomaly detection.
        
        Args:
            metric_name: Name of the metric with anomaly
            entity_id: ID of the entity
            anomaly_type: Type of anomaly
            severity: Severity level
            explanation: Optional explanation
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO anomaly_detections 
                    (metric_name, entity_id, anomaly_type, severity, explanation)
                    VALUES (?, ?, ?, ?, ?)
                """, (metric_name, entity_id, anomaly_type, severity, explanation))
                conn.commit()
                logger.info(f"Anomaly recorded: {anomaly_type} in {metric_name}")
                
        except Exception as e:
            logger.error(f"Failed to record anomaly: {e}")
    
    def get_anomalies(self, resolved: bool = False) -> List[Dict[str, Any]]:
        """
        Get anomaly detections.
        
        Args:
            resolved: Whether to include resolved anomalies
            
        Returns:
            List of anomaly records
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM anomaly_detections"
                if not resolved:
                    query += " WHERE resolved = FALSE"
                query += " ORDER BY detected_at DESC"
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to retrieve anomalies: {e}")
            return []
    
    def log_automation_action(self, action_type: str, entity_id: str, 
                             parameters: Dict[str, Any], dry_run: bool = True,
                             success: bool = True, result_data: Dict[str, Any] = None):
        """
        Log automation action execution.
        
        Args:
            action_type: Type of automation action
            entity_id: ID of the entity acted upon
            parameters: Parameters used for the action
            dry_run: Whether this was a dry run
            success: Whether the action was successful
            result_data: Result data from the action
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO automation_history 
                    (action_type, entity_id, parameters, dry_run, success, result_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    action_type,
                    entity_id,
                    json.dumps(parameters),
                    dry_run,
                    success,
                    json.dumps(result_data) if result_data else None
                ))
                conn.commit()
                logger.info(f"Automation action logged: {action_type}")
                
        except Exception as e:
            logger.error(f"Failed to log automation action: {e}")
    
    def get_automation_history(self, action_type: str = None, 
                              entity_id: str = None) -> List[Dict[str, Any]]:
        """
        Get automation history.
        
        Args:
            action_type: Filter by action type
            entity_id: Filter by entity ID
            
        Returns:
            List of automation history records
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM automation_history WHERE 1=1"
                params = []
                
                if action_type:
                    query += " AND action_type = ?"
                    params.append(action_type)
                if entity_id:
                    query += " AND entity_id = ?"
                    params.append(entity_id)
                
                query += " ORDER BY executed_at DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                history = []
                for row in rows:
                    record = dict(row)
                    record['parameters'] = json.loads(record['parameters'])
                    if record['result_data']:
                        record['result_data'] = json.loads(record['result_data'])
                    history.append(record)
                
                return history
                
        except Exception as e:
            logger.error(f"Failed to retrieve automation history: {e}")
            return []
    
    def cleanup_expired_cache(self):
        """Remove expired cache entries."""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM performance_cache 
                    WHERE expires_at < datetime('now')
                """)
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} expired cache entries")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {e}")
    
    def store_performance_data(self, entity_type: str, entity_id: str, 
                             entity_name: str, data: Dict[str, Any], 
                             source: str = 'facebook_ads'):
        """
        Store performance data for campaigns, adsets, or ads.
        
        Args:
            entity_type: Type of entity ('campaign', 'adset', 'ad')
            entity_id: ID of the entity
            entity_name: Name of the entity
            data: Performance data dictionary
            source: Data source (default: 'facebook_ads')
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Store each metric separately in performance_cache
                for key, value in data.items():
                    if key in ['campaign_id', 'adset_id', 'ad_id', 'campaign_name', 'adset_name', 'ad_name']:
                        continue  # Skip ID and name fields
                    
                    # Try to convert value to float, skip if not numeric
                    try:
                        numeric_value = float(value) if value is not None else 0.0
                    except (ValueError, TypeError):
                        continue
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO performance_cache 
                        (account_id, campaign_id, adset_id, ad_id, date_start, 
                         date_end, metric_name, metric_value, expires_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now', '+24 hours'))
                    """, (
                        data.get('account_id'),
                        data.get('campaign_id') if entity_type in ['campaign', 'adset', 'ad'] else entity_id,
                        data.get('adset_id') if entity_type in ['adset', 'ad'] else None,
                        data.get('ad_id') if entity_type == 'ad' else None,
                        data.get('date_start'),
                        data.get('date_stop'),
                        key,
                        numeric_value
                    ))
                
                conn.commit()
                logger.info(f"Stored performance data for {entity_type} {entity_id}")
                
        except Exception as e:
            logger.error(f"Failed to store performance data: {e}")

    def get_data_summary(self) -> Dict[str, int]:
        """
        Get summary of data stored in the system.
        
        Returns:
            Dictionary with counts of different data types
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                summary = {}
                
                # Count audit entries
                cursor.execute("SELECT COUNT(*) FROM audit_log")
                summary['audit_entries'] = cursor.fetchone()[0]
                
                # Count cached performance data
                cursor.execute("SELECT COUNT(*) FROM performance_cache WHERE expires_at > datetime('now')")
                summary['cached_metrics'] = cursor.fetchone()[0]
                
                # Count AI insights
                cursor.execute("SELECT COUNT(*) FROM ai_insights")
                summary['ai_insights'] = cursor.fetchone()[0]
                
                # Count anomalies
                cursor.execute("SELECT COUNT(*) FROM anomaly_detections WHERE resolved = FALSE")
                summary['active_anomalies'] = cursor.fetchone()[0]
                
                # Count automation actions
                cursor.execute("SELECT COUNT(*) FROM automation_history")
                summary['automation_actions'] = cursor.fetchone()[0]
                
                return summary
                
        except Exception as e:
            logger.error(f"Failed to get data summary: {e}")
            return {}


# Global instance
data_store = DataStore()

# Convenience functions for common operations
def log_audit_entry(action: str, details: Dict[str, Any] = None, 
                   user_id: str = None, success: bool = True, 
                   error_message: str = None):
    """Convenience function for audit logging."""
    return data_store.log_audit_entry(action, details, user_id, success, error_message)

def cache_performance_data(performance_data: List[Dict[str, Any]], 
                          cache_duration_hours: int = 24):
    """Convenience function for caching performance data."""
    return data_store.cache_performance_data(performance_data, cache_duration_hours)

def get_cached_performance_data(filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Convenience function for retrieving cached data."""
    return data_store.get_cached_performance_data(filters)

def store_ai_insight(insight_type: str, entity_id: str, 
                    insight_data: Dict[str, Any], confidence_score: float = 0.0):
    """Convenience function for storing AI insights."""
    return data_store.store_ai_insight(insight_type, entity_id, insight_data, confidence_score)

def record_anomaly(metric_name: str, entity_id: str, 
                  anomaly_type: str, severity: str, explanation: str = None):
    """Convenience function for recording anomalies."""
    return data_store.record_anomaly(metric_name, entity_id, anomaly_type, severity, explanation)

def log_automation_action(action_type: str, entity_id: str, 
                         parameters: Dict[str, Any], dry_run: bool = True,
                         success: bool = True, result_data: Dict[str, Any] = None):
    """Convenience function for logging automation actions."""
    return data_store.log_automation_action(action_type, entity_id, parameters, dry_run, success, result_data)
