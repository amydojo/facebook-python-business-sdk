
"""
Anomaly detection and alert system for marketing metrics.
Detects unusual changes in performance and provides AI-powered explanations.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openai_client import explain_anomaly, call_chat
from data_store import log_audit_entry
from config import config

logger = logging.getLogger(__name__)

def detect_anomalies(df_metrics, metric_col, threshold_pct=0.3, window_days=7):
    """
    Detect anomalies in metrics using statistical thresholds.
    
    Args:
        df_metrics: DataFrame with date and metric columns
        metric_col: name of metric column to analyze
        threshold_pct: percentage threshold for anomaly detection (0.3 = 30%)
        window_days: rolling window for baseline calculation
    
    Returns:
        dict: {
            'flagged': bool,
            'change_percent': float,
            'current_value': float,
            'baseline_value': float,
            'significance': str
        }
    """
    if df_metrics.empty or metric_col not in df_metrics.columns:
        logger.warning(f"No data or missing column {metric_col}")
        return {'flagged': False, 'error': 'No data available'}
    
    try:
        # Ensure data is sorted by date
        if 'date' in df_metrics.columns:
            df = df_metrics.sort_values('date').copy()
        elif 'date_start' in df_metrics.columns:
            df = df_metrics.sort_values('date_start').copy()
            df['date'] = df['date_start']
        else:
            logger.warning("No date column found for time-series analysis")
            return {'flagged': False, 'error': 'No date column found'}
        
        # Get numeric values only
        df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')
        df = df.dropna(subset=[metric_col])
        
        if len(df) < 2:
            logger.warning(f"Insufficient data points for anomaly detection: {len(df)}")
            return {'flagged': False, 'error': 'Insufficient data points'}
        
        # Get most recent value
        current_value = df[metric_col].iloc[-1]
        
        # Calculate baseline (mean of previous values)
        if len(df) >= window_days:
            baseline_values = df[metric_col].iloc[-window_days:-1]
        else:
            baseline_values = df[metric_col].iloc[:-1]
        
        if len(baseline_values) == 0:
            return {'flagged': False, 'error': 'No baseline data available'}
        
        baseline_mean = baseline_values.mean()
        baseline_std = baseline_values.std()
        
        # Calculate percentage change
        if baseline_mean != 0:
            change_percent = ((current_value - baseline_mean) / baseline_mean) * 100
        else:
            change_percent = 0
        
        # Determine if anomaly
        flagged = abs(change_percent) > (threshold_pct * 100)
        
        # Determine significance level
        if abs(change_percent) > 50:
            significance = 'critical'
        elif abs(change_percent) > 30:
            significance = 'high'
        elif abs(change_percent) > 15:
            significance = 'medium'
        else:
            significance = 'low'
        
        result = {
            'flagged': flagged,
            'change_percent': change_percent,
            'current_value': current_value,
            'baseline_value': baseline_mean,
            'baseline_std': baseline_std,
            'significance': significance,
            'metric_name': metric_col,
            'data_points': len(df),
            'window_days': window_days
        }
        
        if flagged:
            logger.warning(f"🚨 Anomaly detected in {metric_col}: {change_percent:+.1f}% change")
        else:
            logger.info(f"✅ No anomaly in {metric_col}: {change_percent:+.1f}% change within threshold")
        
        return result
        
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        return {'flagged': False, 'error': str(e)}

def analyze_and_alert(df_metrics, metric_name, context="", send_alert=True):
    """
    Analyze anomalies and generate AI-powered explanations and alerts.
    
    Args:
        df_metrics: DataFrame with metrics data
        metric_name: name of metric to analyze
        context: additional context for analysis
        send_alert: whether to send alerts for detected anomalies
    
    Returns:
        dict: analysis results with AI explanation
    """
    try:
        # Detect anomalies
        anomaly_result = detect_anomalies(df_metrics, metric_name)
        
        if not anomaly_result['flagged']:
            logger.info(f"No anomalies detected in {metric_name}")
            return {
                'anomaly_detected': False,
                'metric_name': metric_name,
                'analysis': anomaly_result
            }
        
        # Generate AI explanation for the anomaly
        ai_explanation = explain_anomaly(
            metric_name=anomaly_result['metric_name'],
            change_percent=anomaly_result['change_percent'],
            current_value=anomaly_result['current_value'],
            previous_value=anomaly_result['baseline_value'],
            context=context
        )
        
        # Prepare alert data
        alert_data = {
            'anomaly_detected': True,
            'metric_name': metric_name,
            'change_percent': anomaly_result['change_percent'],
            'current_value': anomaly_result['current_value'],
            'baseline_value': anomaly_result['baseline_value'],
            'significance': anomaly_result['significance'],
            'ai_explanation': ai_explanation,
            'timestamp': datetime.now(),
            'context': context
        }
        
        # Log to audit trail
        log_audit_entry({
            'action_type': 'anomaly_detection',
            'metric_name': metric_name,
            'anomaly_data': alert_data,
            'timestamp': datetime.now()
        })
        
        # Send alert if requested
        if send_alert:
            alert_sent = send_anomaly_alert(alert_data)
            alert_data['alert_sent'] = alert_sent
        
        logger.info(f"🔍 Anomaly analysis complete for {metric_name}")
        return alert_data
        
    except Exception as e:
        logger.error(f"Error in analyze_and_alert: {e}")
        return {
            'anomaly_detected': False,
            'error': str(e),
            'metric_name': metric_name
        }

def send_anomaly_alert(alert_data):
    """
    Send anomaly alert via configured channels (Slack, email, etc.).
    
    Args:
        alert_data: dict with anomaly information
    
    Returns:
        bool: True if alert sent successfully
    """
    try:
        # Format alert message
        message = format_alert_message(alert_data)
        
        # Log the alert (placeholder for actual alert sending)
        logger.warning(f"📢 ANOMALY ALERT: {message}")
        
        # TODO: Implement actual alert sending
        # - Slack webhook
        # - Email notification
        # - SMS alert
        # For now, just log the alert
        
        return True
        
    except Exception as e:
        logger.error(f"Error sending anomaly alert: {e}")
        return False

def format_alert_message(alert_data):
    """
    Format anomaly alert message for notifications.
    
    Args:
        alert_data: dict with anomaly information
    
    Returns:
        str: formatted alert message
    """
    try:
        metric_name = alert_data['metric_name']
        change_pct = alert_data['change_percent']
        current_val = alert_data['current_value']
        significance = alert_data['significance']
        
        direction = "increased" if change_pct > 0 else "decreased"
        icon = "📈" if change_pct > 0 else "📉"
        
        message = f"""
🚨 MARKETING ANOMALY DETECTED

{icon} Metric: {metric_name}
📊 Change: {direction} by {abs(change_pct):.1f}%
📋 Current Value: {current_val:,.2f}
⚠️ Significance: {significance.upper()}

🤖 AI Analysis:
{alert_data.get('ai_explanation', 'Analysis not available')}

🕒 Detected at: {alert_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()
        
        return message
        
    except Exception as e:
        logger.error(f"Error formatting alert message: {e}")
        return f"Anomaly detected in {alert_data.get('metric_name', 'unknown metric')}"

def batch_anomaly_detection(metrics_data_dict, thresholds=None):
    """
    Run anomaly detection on multiple metrics.
    
    Args:
        metrics_data_dict: dict {metric_name: DataFrame}
        thresholds: dict {metric_name: threshold_pct} or None for defaults
    
    Returns:
        dict: {metric_name: anomaly_result}
    """
    if thresholds is None:
        thresholds = {}
    
    results = {}
    
    try:
        for metric_name, df in metrics_data_dict.items():
            threshold = thresholds.get(metric_name, 0.3)  # Default 30%
            
            logger.info(f"Running anomaly detection for {metric_name}")
            result = analyze_and_alert(df, metric_name, send_alert=False)
            results[metric_name] = result
        
        # Count anomalies
        anomaly_count = sum(1 for r in results.values() if r.get('anomaly_detected', False))
        
        logger.info(f"Batch anomaly detection complete: {anomaly_count}/{len(results)} metrics flagged")
        
        # Send summary if multiple anomalies detected
        if anomaly_count > 1:
            send_batch_anomaly_summary(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch anomaly detection: {e}")
        return {}

def send_batch_anomaly_summary(anomaly_results):
    """
    Send summary alert for multiple detected anomalies.
    
    Args:
        anomaly_results: dict from batch_anomaly_detection()
    """
    try:
        flagged_metrics = {
            name: result for name, result in anomaly_results.items()
            if result.get('anomaly_detected', False)
        }
        
        if not flagged_metrics:
            return
        
        # Generate AI summary of multiple anomalies
        summary_prompt = f"""
        Multiple marketing metrics have shown anomalous behavior:
        
        {chr(10).join([f"- {name}: {result['change_percent']:+.1f}% change" for name, result in flagged_metrics.items()])}
        
        Provide a brief analysis of what this pattern might indicate and suggest immediate actions to investigate.
        """
        
        messages = [
            {"role": "system", "content": "You are a marketing performance analyst specializing in anomaly investigation."},
            {"role": "user", "content": summary_prompt}
        ]
        
        ai_summary = call_chat(messages, max_tokens=200, temperature=0.7)
        
        summary_message = f"""
🚨 MULTIPLE ANOMALIES DETECTED

📊 {len(flagged_metrics)} metrics showing unusual behavior:
{chr(10).join([f"• {name}: {result['change_percent']:+.1f}%" for name, result in flagged_metrics.items()])}

🤖 AI Analysis:
{ai_summary or 'Analysis not available'}

🔍 Immediate Action Required: Review campaign performance and investigate potential causes.
        """
        
        logger.warning(f"📢 BATCH ANOMALY SUMMARY: {summary_message}")
        
        # Log summary
        log_audit_entry({
            'action_type': 'batch_anomaly_summary',
            'anomaly_count': len(flagged_metrics),
            'summary': summary_message,
            'timestamp': datetime.now()
        })
        
    except Exception as e:
        logger.error(f"Error sending batch anomaly summary: {e}")

def schedule_daily_anomaly_check():
    """
    Run daily anomaly check on key metrics.
    This would be called by a scheduler or webhook.
    
    Returns:
        dict: results of daily check
    """
    try:
        logger.info("🕒 Starting daily anomaly check")
        
        # This would integrate with fetch_paid.py and fetch_organic.py
        # to get fresh data for analysis
        
        # Placeholder implementation
        results = {
            'check_date': datetime.now().date(),
            'metrics_checked': 0,
            'anomalies_detected': 0,
            'alerts_sent': 0,
            'status': 'completed'
        }
        
        logger.info("📊 Daily anomaly check completed")
        
        # Log daily check
        log_audit_entry({
            'action_type': 'daily_anomaly_check',
            'results': results,
            'timestamp': datetime.now()
        })
        
        return results
        
    except Exception as e:
        logger.error(f"Error in daily anomaly check: {e}")
        return {
            'check_date': datetime.now().date(),
            'status': 'failed',
            'error': str(e)
        }

def get_anomaly_detection_config():
    """
    Get default anomaly detection configuration.
    
    Returns:
        dict: configuration for different metric types
    """
    return {
        'spend': {'threshold_pct': 0.25, 'window_days': 7},
        'impressions': {'threshold_pct': 0.30, 'window_days': 7},
        'clicks': {'threshold_pct': 0.35, 'window_days': 7},
        'ctr': {'threshold_pct': 0.20, 'window_days': 7},
        'cpc': {'threshold_pct': 0.25, 'window_days': 7},
        'conversions': {'threshold_pct': 0.40, 'window_days': 7},
        'roas': {'threshold_pct': 0.20, 'window_days': 7},
        'reach': {'threshold_pct': 0.30, 'window_days': 7},
        'frequency': {'threshold_pct': 0.15, 'window_days': 7}
    }

def simulate_anomaly_detection(metric_name, days_back=14):
    """
    Simulate anomaly detection with synthetic data for testing.
    
    Args:
        metric_name: name of metric to simulate
        days_back: number of days of data to generate
    
    Returns:
        dict: simulated anomaly detection results
    """
    try:
        # Generate synthetic data with an anomaly
        dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
        
        # Normal values with some noise
        normal_mean = 1000
        normal_values = np.random.normal(normal_mean, normal_mean * 0.1, days_back-1)
        
        # Add an anomaly in the last day
        anomaly_value = normal_mean * (1.5 if np.random.random() > 0.5 else 0.5)  # +50% or -50%
        
        values = np.append(normal_values, anomaly_value)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            metric_name: values
        })
        
        # Run anomaly detection
        result = detect_anomalies(df, metric_name)
        
        logger.info(f"🎭 Simulated anomaly detection for {metric_name}: {result}")
        
        return {
            'simulated': True,
            'metric_name': metric_name,
            'synthetic_data': df,
            'anomaly_result': result
        }
        
    except Exception as e:
        logger.error(f"Error in anomaly simulation: {e}")
        return {'simulated': True, 'error': str(e)}
