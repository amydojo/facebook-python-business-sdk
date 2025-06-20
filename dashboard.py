
"""
Streamlit dashboard for AI-powered social campaign optimizer.
Official docs: https://docs.streamlit.io/
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="AI Campaign Optimizer",
    page_icon="🎯",
    layout="wide"
)

def check_environment():
    """
    Check required environment variables and Facebook SDK imports.
    Returns dict with status and missing items.
    """
    # Check required environment variables
    required_env = ["META_ACCESS_TOKEN", "AD_ACCOUNT_ID"]
    missing_env = [k for k in required_env if not os.getenv(k)]
    
    # Check optional environment variables
    optional_env = ["META_APP_ID", "META_APP_SECRET", "PAGE_ID"]
    missing_optional = [k for k in optional_env if not os.getenv(k)]
    
    if missing_env:
        logger.error(f"❌ Missing required environment variables: {missing_env}")
    if missing_optional:
        logger.warning(f"⚠️ Missing optional environment variables: {missing_optional}")
    
    # Check Facebook SDK imports
    sdk_status = {"available": False, "error": None}
    try:
        import facebook_business.api as fb_api
        logger.info(f"✅ facebook_business.api location: {fb_api.__file__}")
        sdk_status["available"] = True
    except Exception as e:
        logger.error(f"❌ Failed to import facebook_business.api: {e}", exc_info=True)
        sdk_status["error"] = str(e)
    
    return {
        "missing_env": missing_env,
        "missing_optional": missing_optional,
        "sdk_status": sdk_status
    }

def main():
    logger.info("🚀 Starting Streamlit app")
    
    # Log organic insights environment status
    page_token_set = bool(os.getenv('PAGE_ACCESS_TOKEN'))
    page_id = os.getenv('PAGE_ID')
    ig_user_id = os.getenv('IG_USER_ID')
    
    logger.info(f"Using PAGE_ACCESS_TOKEN set: {page_token_set}, PAGE_ID: {page_id}")
    logger.info(f"IG_USER_ID set: {bool(ig_user_id)}")
    
    # Environment and SDK checks
    env_check = check_environment()
    
    st.title("🎯 AI-Powered Social Campaign Optimizer")
    st.markdown("Minimize manual work, maximize ad performance and organic engagement")
    
    # Show environment status
    if env_check["missing_env"]:
        st.error(f"❌ Missing required environment variables: {env_check['missing_env']}")
        st.info("Please configure these variables in Replit Secrets to continue.")
        st.stop()
    
    if not env_check["sdk_status"]["available"]:
        st.error(f"❌ Facebook Business SDK not available: {env_check['sdk_status']['error']}")
        st.info("Please check the installation and remove any local facebook_business modules.")
        st.stop()
    
    # Import modules after environment check
    try:
        from data_store import data_store
        from fb_client import fb_client
        from fetch_paid import get_campaign_performance, get_campaign_performance_summary
        from fetch_organic import get_organic_insights, get_organic_performance_summary, fetch_latest_ig_media_insights
        from anomaly import detect_anomalies
        from auto_actions import validate_automation_safety
        from config import config
    except ImportError as e:
        st.error(f"❌ Failed to import modules: {e}")
        logger.error(f"❌ Module import error: {e}", exc_info=True)
        st.stop()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Check API connection
        if fb_client.is_initialized():
            st.success("✅ Facebook API Connected")
            
            # Test connection
            if st.button("Test Connection"):
                with st.spinner("Testing connection..."):
                    test_result = fb_client.test_connection()
                    if test_result["success"]:
                        st.success(f"✅ Connection successful!")
                        st.json(test_result)
                    else:
                        st.error(f"❌ Connection failed: {test_result['error']}")
        else:
            st.error("❌ Facebook API Not Connected")
            st.info("Check your Meta credentials in Replit Secrets")
        
        # Show environment variables status
        st.subheader("Environment Status")
        env_vars = {
            "META_ACCESS_TOKEN": bool(os.getenv("META_ACCESS_TOKEN")),
            "AD_ACCOUNT_ID": bool(os.getenv("AD_ACCOUNT_ID")),
            "META_APP_ID": bool(os.getenv("META_APP_ID")),
            "META_APP_SECRET": bool(os.getenv("META_APP_SECRET")),
            "PAGE_ACCESS_TOKEN": bool(os.getenv("PAGE_ACCESS_TOKEN")),
            "PAGE_ID": bool(os.getenv("PAGE_ID")),
            "IG_USER_ID": bool(os.getenv("IG_USER_ID"))
        }
        
        for var, is_set in env_vars.items():
            if is_set:
                st.success(f"✅ {var}")
            else:
                st.warning(f"⚠️ {var}")
        
        # Automation safety check
        try:
            safety_status = validate_automation_safety()
            if safety_status.get('safe_to_automate'):
                st.success("✅ Automation Safety Validated")
            else:
                st.warning("⚠️ Automation Safety Check Failed")
        except Exception as e:
            st.warning(f"⚠️ Could not check automation safety: {e}")
        
        # Data store status
        try:
            data_summary = data_store.get_data_summary()
            st.subheader("Data Store Status")
            for key, value in data_summary.items():
                st.metric(key.replace('_', ' ').title(), value)
        except Exception as e:
            st.warning(f"⚠️ Could not get data store status: {e}")
    
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
            
            # Date preset selector
            date_preset = st.selectbox(
                "Select time range:",
                ["yesterday", "last_7d", "last_30d", "this_month", "last_month"],
                index=1
            )
            
            if st.button("Fetch Latest Paid Data"):
                with st.spinner("Fetching campaign data..."):
                    try:
                        paid_data = get_campaign_performance(date_preset=date_preset)
                        if not paid_data.empty:
                            st.success(f"✅ Fetched {len(paid_data)} campaign records")
                            
                            # Display summary metrics
                            summary = get_campaign_performance_summary(date_preset=date_preset)
                            if summary:
                                col_a, col_b, col_c, col_d = st.columns(4)
                                with col_a:
                                    st.metric("Total Spend", f"${summary['total_spend']:.2f}")
                                with col_b:
                                    st.metric("Total Impressions", f"{summary['total_impressions']:,}")
                                with col_c:
                                    st.metric("Total Clicks", f"{summary['total_clicks']:,}")
                                with col_d:
                                    st.metric("Avg CTR", f"{summary['average_ctr']:.2f}%")
                            
                            # Display data table
                            st.dataframe(paid_data)
                            
                            # Simple chart
                            if 'spend' in paid_data.columns and not paid_data['spend'].isna().all():
                                fig = px.bar(paid_data, x='campaign_name', y='spend', 
                                           title='Campaign Spend')
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No paid campaign data available for the selected time range")
                    except Exception as e:
                        st.error(f"Error fetching paid data: {e}")
                        logger.error(f"❌ Error fetching paid data: {e}", exc_info=True)
        
        with col2:
            st.subheader("Organic Content")
            
            # Check organic insights environment
            try:
                from fetch_organic import (
                    validate_organic_environment, ORGANIC_DATE_PRESETS, 
                    get_available_page_metrics, get_valid_instagram_metrics,
                    fetch_latest_ig_media_insights
                )
                organic_validation = validate_organic_environment()
                
                # Show detailed environment status with actionable warnings
                if not organic_validation['page_insights_enabled']:
                    st.error("❌ Facebook Page insights disabled")
                    if not organic_validation['page_token_available']:
                        st.info("💡 Add PAGE_ACCESS_TOKEN to Replit Secrets to enable Page insights")
                    if not organic_validation['page_id_available']:
                        st.info("💡 Add PAGE_ID to Replit Secrets to enable Page insights")
                else:
                    st.success("✅ Facebook Page insights enabled")
                
                if organic_validation['ig_user_id_available']:
                    if organic_validation['instagram_insights_enabled']:
                        st.success("✅ Instagram insights enabled")
                    else:
                        st.warning("⚠️ IG_USER_ID set but PAGE_ACCESS_TOKEN missing—Instagram insights disabled")
                else:
                    st.info("ℹ️ Add IG_USER_ID to Replit Secrets to enable Instagram insights")
                
                # Show available metrics info
                with st.expander("📊 Available Metrics Info"):
                    col_page_metrics, col_ig_metrics = st.columns(2)
                    
                    with col_page_metrics:
                        st.subheader("📘 Page Metrics")
                        if organic_validation['page_insights_enabled']:
                            available_page_metrics = get_available_page_metrics()
                            if available_page_metrics:
                                st.success(f"✅ {len(available_page_metrics)} metrics available")
                                st.text("\n".join(available_page_metrics))
                            else:
                                st.warning("⚠️ Could not fetch available metrics")
                        else:
                            st.info("Enable Page insights to see available metrics")
                    
                    with col_ig_metrics:
                        st.subheader("📸 Instagram Metrics")
                        valid_ig_metrics = get_valid_instagram_metrics()
                        st.success(f"✅ {len(valid_ig_metrics)} valid metrics")
                        st.text("\n".join(valid_ig_metrics[:10]))  # Show first 10
                        if len(valid_ig_metrics) > 10:
                            st.text("... and more")
                
                # Enhanced date preset selector
                organic_date_preset = st.selectbox(
                    "Select organic time range:",
                    list(ORGANIC_DATE_PRESETS.keys()),
                    format_func=lambda x: ORGANIC_DATE_PRESETS[x],
                    index=1,  # Default to "yesterday"
                    key="organic_preset"
                )
                
                # Custom date range option
                custom_since = None
                custom_until = None
                if organic_date_preset == "custom":
                    st.subheader("📅 Custom Date Range")
                    col_since, col_until = st.columns(2)
                    with col_since:
                        custom_since = st.date_input("Start Date", value=date.today() - timedelta(days=7))
                    with col_until:
                        custom_until = st.date_input("End Date", value=date.today() - timedelta(days=1))
                    
                    if custom_since and custom_until:
                        if custom_since > custom_until:
                            st.error("❌ Start date must be before end date")
                            custom_since = custom_until = None
                
                col_fetch, col_instagram = st.columns([1, 1])
                
                with col_fetch:
                    if st.button("Fetch Organic Data"):
                        with st.spinner("Fetching organic insights..."):
                            try:
                                # Prepare parameters for organic insights fetch
                                fetch_params = {'include_instagram': True}
                                
                                if organic_date_preset == "custom" and custom_since and custom_until:
                                    fetch_params.update({
                                        'since': custom_since.strftime('%Y-%m-%d'),
                                        'until': custom_until.strftime('%Y-%m-%d')
                                    })
                                    logger.info(f"Fetching organic data for custom range: {custom_since} to {custom_until}")
                                else:
                                    fetch_params['date_preset'] = organic_date_preset
                                    logger.info(f"Fetching organic data with preset: {organic_date_preset}")
                                
                                organic_data = get_organic_insights(**fetch_params)
                                
                                if not organic_data.empty:
                                    st.success(f"✅ Fetched {len(organic_data)} organic insights records")
                                    
                                    # Display summary metrics for latest data
                                    if organic_date_preset in ['latest', 'yesterday']:
                                        summary = get_organic_performance_summary(organic_date_preset)
                                        if summary:
                                            col_a, col_b, col_c, col_d = st.columns(4)
                                            with col_a:
                                                st.metric("Total Reach", f"{summary['total_reach']:,}")
                                            with col_b:
                                                st.metric("Total Impressions", f"{summary['total_impressions']:,}")
                                            with col_c:
                                                st.metric("Total Engagement", f"{summary['total_engagement']:,}")
                                            with col_d:
                                                st.metric("Engagement Rate", f"{summary['avg_engagement_rate']:.2f}%")
                                    
                                    # Separate Page and Instagram data
                                    if 'source' in organic_data.columns:
                                        page_data = organic_data[organic_data['source'] == 'facebook_page']
                                        ig_data = organic_data[organic_data['source'] == 'instagram']
                                        
                                        if not page_data.empty:
                                            st.subheader("📘 Facebook Page Insights")
                                            st.dataframe(page_data)
                                            
                                            # Chart for page metrics
                                            page_reach_data = page_data[page_data['metric'] == 'page_reach']
                                            if not page_reach_data.empty:
                                                fig = px.line(page_reach_data, x='date', y='value', 
                                                            title='Facebook Page Reach Over Time')
                                                st.plotly_chart(fig, use_container_width=True)
                                        
                                        if not ig_data.empty:
                                            st.subheader("📸 Instagram Insights")
                                            st.dataframe(ig_data)
                                            
                                            # Chart for Instagram metrics
                                            ig_impressions = ig_data[ig_data['metric'] == 'impressions']
                                            if not ig_impressions.empty:
                                                fig = px.bar(ig_impressions, x='date', y='value',
                                                           title='Instagram Impressions by Media')
                                                st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.dataframe(organic_data)
                                        
                                        # Simple chart for organic reach
                                        if 'page_reach' in organic_data.columns:
                                            fig = px.line(organic_data, x='date', y='page_reach', 
                                                        title='Page Reach Over Time')
                                            st.plotly_chart(fig, use_container_width=True)
                                else:
                                    # Provide specific guidance based on date preset and environment
                                    if organic_date_preset in ['latest', 'yesterday']:
                                        st.warning("⚠️ No data for latest organic insights")
                                        st.info("Possible causes:")
                                        st.info("• No Facebook Page or Instagram activity yesterday")
                                        st.info("• Token permissions don't include insights")
                                        st.info("• Page/Instagram account not properly linked")
                                    elif organic_date_preset == "custom":
                                        st.info(f"No organic data for {custom_since} to {custom_until}")
                                    else:
                                        st.info("No organic data available for the selected time range")
                                    
                                    # Show validation status for debugging
                                    st.subheader("🔍 Troubleshooting Info")
                                    st.json(organic_validation)
                                        
                            except Exception as e:
                                st.error(f"❌ Error fetching organic data: {e}")
                                logger.error(f"❌ Error fetching organic data: {e}", exc_info=True)
                                
                                # Provide specific troubleshooting guidance
                                st.subheader("🛠️ Troubleshooting Steps")
                                st.info("1. Check if PAGE_ACCESS_TOKEN has sufficient permissions")
                                st.info("2. Verify PAGE_ID and IG_USER_ID are correct")
                                st.info("3. Ensure the Facebook Page and Instagram account are properly linked")
                                st.info("4. Check if the selected date range has any activity")
                                st.info("5. Review the logs for specific API error messages")
                
                with col_instagram:
                    # Instagram-specific latest insights
                    if organic_validation['instagram_insights_enabled'] and st.button("Latest Instagram Only"):
                        with st.spinner("Fetching latest Instagram insights..."):
                            try:
                                from fetch_organic import fetch_latest_ig_media_insights
                                ig_user_id = os.getenv('IG_USER_ID')
                                
                                ig_data = fetch_latest_ig_media_insights(
                                    ig_user_id, 
                                    metrics=['impressions', 'reach', 'engagement']
                                )
                                
                                if not ig_data.empty:
                                    st.success(f"✅ Fetched {len(ig_data)} Instagram insights")
                                    st.subheader("📸 Latest Instagram Media Insights")
                                    st.dataframe(ig_data)
                                else:
                                    st.warning("⚠️ No Instagram media insights for yesterday. Check if any posts were made.")
                                    
                            except Exception as e:
                                st.error(f"Error fetching Instagram data: {e}")
                                logger.error(f"❌ Error fetching Instagram data: {e}", exc_info=True)
                
            except ImportError as e:
                st.error(f"❌ Failed to import organic insights modules: {e}")
                logger.error(f"❌ Organic insights module import error: {e}", exc_info=True)
    
    with tab2:
        st.header("Anomaly Detection")
        
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
                    logger.error(f"❌ Anomaly detection error: {e}", exc_info=True)
    
    with tab3:
        st.header("AI Insights")
        st.info("AI insights feature will be implemented based on collected data patterns.")
    
    with tab4:
        st.header("Automation Management")
        st.info("Automation controls will be implemented after gathering sufficient performance data.")
    
    with tab5:
        st.header("Audit Log")
        st.info("Audit logging will track all automated actions and manual interventions.")
    
    # Footer
    st.markdown("---")
    st.markdown("🚀 AI-Powered Campaign Optimizer - Built with ❤️ on Replit")
    st.markdown("**Official Documentation:**")
    st.markdown("- [Facebook Business SDK](https://developers.facebook.com/docs/business-sdk/)")
    st.markdown("- [Marketing API Insights](https://developers.facebook.com/docs/marketing-api/insights/)")
    st.markdown("- [Streamlit Documentation](https://docs.streamlit.io/)")

if __name__ == "__main__":
    main()
