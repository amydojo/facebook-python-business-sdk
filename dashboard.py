
"""
Streamlit dashboard for AI-powered social campaign optimizer.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Import our modules
from data_store import data_store
from fb_client import fb_client
from fetch_paid import get_campaign_performance
from fetch_organic import get_organic_insights
from anomaly import detect_anomalies
from auto_actions import validate_automation_safety
from config import config

# Configure page
st.set_page_config(
    page_title="AI Campaign Optimizer",
    page_icon="🎯",
    layout="wide"
)

def main():
    st.title("🎯 AI-Powered Social Campaign Optimizer")
    st.markdown("Minimize manual work, maximize ad performance and organic engagement")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Check API connection
        if fb_client.is_initialized():
            st.success("✅ Facebook API Connected")
        else:
            st.error("❌ Facebook API Not Connected")
            st.info("Please configure your Meta credentials in Replit Secrets")
        
        # Automation safety check
        safety_status = validate_automation_safety()
        if safety_status.get('safe_to_automate'):
            st.success("✅ Automation Safety Validated")
        else:
            st.warning("⚠️ Automation Safety Check Failed")
        
        # Data store status
        data_summary = data_store.get_data_summary()
        st.subheader("Data Store Status")
        for key, value in data_summary.items():
            st.metric(key.replace('_', ' ').title(), value)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Performance Overview", 
        "🔍 Anomaly Detection", 
        "🤖 AI Insights", 
        "⚙️ Automation", 
        "📝 Audit Log"
    ])
    
    with tab1:
        st.header("Performance Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Paid Campaigns")
            if st.button("Fetch Latest Paid Data"):
                with st.spinner("Fetching campaign data..."):
                    try:
                        paid_data = get_campaign_performance()
                        if paid_data:
                            st.success(f"✅ Fetched {len(paid_data)} campaign records")
                            df = pd.DataFrame(paid_data)
                            st.dataframe(df)
                        else:
                            st.info("No paid campaign data available")
                    except Exception as e:
                        st.error(f"Error fetching paid data: {e}")
            
            # Show cached data
            cached_data = data_store.get_cached_performance_data()
            if cached_data:
                st.subheader("Cached Performance Data")
                df_cached = pd.DataFrame(cached_data)
                st.dataframe(df_cached)
        
        with col2:
            st.subheader("Organic Content")
            if st.button("Fetch Latest Organic Data"):
                with st.spinner("Fetching organic insights..."):
                    try:
                        organic_data = get_organic_insights()
                        if organic_data:
                            st.success(f"✅ Fetched organic insights")
                            st.json(organic_data)
                        else:
                            st.info("No organic data available")
                    except Exception as e:
                        st.error(f"Error fetching organic data: {e}")
    
    with tab2:
        st.header("Anomaly Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("Run Anomaly Detection"):
                with st.spinner("Detecting anomalies..."):
                    try:
                        # Get sample data for anomaly detection
                        sample_data = pd.DataFrame({
                            'date': pd.date_range(start='2024-01-01', periods=30, freq='D'),
                            'impressions': [1000 + i*50 + (i%7)*200 for i in range(30)],
                            'clicks': [50 + i*2 + (i%7)*10 for i in range(30)],
                            'spend': [100 + i*5 + (i%7)*20 for i in range(30)]
                        })
                        
                        anomalies = detect_anomalies(sample_data, 'impressions')
                        
                        if anomalies:
                            st.success(f"✅ Detected {len(anomalies)} anomalies")
                            for anomaly in anomalies:
                                st.warning(f"🚨 {anomaly}")
                        else:
                            st.info("No anomalies detected")
                            
                    except Exception as e:
                        st.error(f"Error in anomaly detection: {e}")
        
        with col2:
            st.subheader("Active Anomalies")
            anomalies = data_store.get_anomalies()
            if anomalies:
                for anomaly in anomalies[:5]:  # Show top 5
                    st.warning(f"⚠️ {anomaly['anomaly_type']} in {anomaly['metric_name']}")
            else:
                st.info("No active anomalies")
    
    with tab3:
        st.header("AI Insights")
        
        insights = data_store.get_ai_insights()
        
        if insights:
            for insight in insights[:10]:  # Show top 10
                with st.expander(f"{insight['insight_type']} - {insight['entity_id']}"):
                    st.write(f"**Created:** {insight['created_at']}")
                    st.write(f"**Confidence:** {insight['confidence_score']:.2f}")
                    st.json(insight['insight_data'])
        else:
            st.info("No AI insights available yet")
    
    with tab4:
        st.header("Automation Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Automation Controls")
            
            # Feature flag control
            automation_enabled = st.checkbox(
                "Enable Automation",
                value=config.ENABLE_AUTO_ACTIONS,
                help="Enable or disable automated actions"
            )
            
            if st.button("Test Automation Safety"):
                safety_check = validate_automation_safety()
                st.json(safety_check)
            
            # Dry run automation
            if st.button("Run Automation (Dry Run)"):
                st.info("🔍 Dry run mode - no actual changes will be made")
                # This would call your automation functions in dry run mode
        
        with col2:
            st.subheader("Automation History")
            history = data_store.get_automation_history()
            if history:
                df_history = pd.DataFrame(history[:10])  # Show recent 10
                st.dataframe(df_history[['action_type', 'entity_id', 'executed_at', 'dry_run', 'success']])
            else:
                st.info("No automation history available")
    
    with tab5:
        st.header("Audit Log")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            date_filter = st.date_input("From Date", datetime.now() - timedelta(days=7))
        with col2:
            action_filter = st.text_input("Filter by Action")
        
        # Get audit entries
        try:
            with data_store.get_db_connection() as conn:
                query = """
                    SELECT timestamp, action, details, user_id, success, error_message
                    FROM audit_log 
                    WHERE timestamp >= ?
                """
                params = [date_filter]
                
                if action_filter:
                    query += " AND action LIKE ?"
                    params.append(f"%{action_filter}%")
                
                query += " ORDER BY timestamp DESC LIMIT 100"
                
                df_audit = pd.read_sql_query(query, conn, params=params)
                
                if not df_audit.empty:
                    st.dataframe(df_audit)
                else:
                    st.info("No audit entries found for the selected criteria")
                    
        except Exception as e:
            st.error(f"Error loading audit log: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("🚀 AI-Powered Campaign Optimizer - Built with ❤️ on Replit")

if __name__ == "__main__":
    main()
