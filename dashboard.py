"""Streamlit dashboard updated to handle long-format Instagram insights."""
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

            # Option to include creative previews
            include_creatives = st.checkbox("Include Creative Previews", value=True)

            if st.button("Fetch Latest Paid Data"):
                with st.spinner("Fetching campaign data..."):
                    try:
                        from fetch_paid import get_campaign_performance_with_creatives
                        paid_data = get_campaign_performance_with_creatives(
                            date_preset=date_preset, 
                            include_creatives=include_creatives
                        )
                        
                        if not paid_data.empty:
                            st.success(f"✅ Fetched {len(paid_data)} campaign records")

                            # Display summary metrics
                            total_spend = paid_data['spend'].sum() if 'spend' in paid_data.columns else 0
                            total_impr = paid_data['impressions'].sum() if 'impressions' in paid_data.columns else 0
                            total_clicks = paid_data['clicks'].sum() if 'clicks' in paid_data.columns else 0
                            avg_ctr = (total_clicks / total_impr * 100) if total_impr > 0 else 0

                            col_a, col_b, col_c, col_d = st.columns(4)
                            with col_a:
                                st.metric("Total Spend", f"${total_spend:.2f}")
                            with col_b:
                                st.metric("Total Impressions", f"{int(total_impr):,}")
                            with col_c:
                                st.metric("Total Clicks", f"{int(total_clicks):,}")
                            with col_d:
                                st.metric("Avg CTR", f"{avg_ctr:.2f}%")

                            # Campaign spend chart with readable labels
                            if 'spend' in paid_data.columns and 'campaign_name' in paid_data.columns:
                                campaign_sums = paid_data.groupby('campaign_name')['spend'].sum().reset_index()
                                if not campaign_sums.empty:
                                    fig = go.Figure(data=[
                                        go.Bar(
                                            x=campaign_sums['campaign_name'],
                                            y=campaign_sums['spend'],
                                            text=[f"${x:.0f}" for x in campaign_sums['spend']],
                                            textposition='auto',
                                        )
                                    ])
                                    fig.update_layout(
                                        title="Campaign Spend",
                                        xaxis_title="Campaign",
                                        yaxis_title="Spend ($)",
                                        xaxis_tickangle=-45,
                                        height=400
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                            # Show creative previews if available
                            if include_creatives and 'creative_image_url' in paid_data.columns:
                                st.markdown("---")
                                st.subheader("🎨 Ad Creative Previews")
                                
                                # Group by campaign for better organization
                                for campaign_name in paid_data['campaign_name'].unique():
                                    if pd.notna(campaign_name):
                                        campaign_data = paid_data[paid_data['campaign_name'] == campaign_name]
                                        
                                        with st.expander(f"📊 {campaign_name}", expanded=True):
                                            for _, row in campaign_data.iterrows():
                                                if pd.notna(row.get('ad_name')):
                                                    col_img, col_details = st.columns([1, 2])
                                                    
                                                    with col_img:
                                                        img_url = row.get('creative_image_url') or row.get('creative_thumbnail_url')
                                                        if img_url and pd.notna(img_url):
                                                            try:
                                                                st.image(img_url, width=200, caption="Creative Preview")
                                                            except Exception as e:
                                                                st.warning(f"Could not load image: {e}")
                                                        else:
                                                            st.info("No preview image available")
                                                    
                                                    with col_details:
                                                        st.markdown(f"**Ad:** {row.get('ad_name', 'Unknown')}")
                                                        
                                                        if pd.notna(row.get('creative_title')):
                                                            st.markdown(f"**Title:** {row['creative_title']}")
                                                        
                                                        if pd.notna(row.get('creative_body')):
                                                            st.write(f"**Text:** {row['creative_body']}")
                                                        
                                                        # Performance metrics with better formatting
                                                        metrics_cols = st.columns(4)
                                                        with metrics_cols[0]:
                                                            st.metric("Impressions", f"{int(row.get('impressions', 0)):,}")
                                                        with metrics_cols[1]:
                                                            st.metric("Clicks", f"{int(row.get('clicks', 0)):,}")
                                                        with metrics_cols[2]:
                                                            st.metric("Spend", f"${row.get('spend', 0):.2f}")
                                                        with metrics_cols[3]:
                                                            ctr = row.get('ctr', 0)
                                                            st.metric("CTR", f"{ctr:.2f}%" if ctr else "0%")
                                                        
                                                        if pd.notna(row.get('creative_object_url')):
                                                            st.markdown(f"🔗 [View Ad Destination]({row['creative_object_url']})")
                                                    
                                                    st.markdown("---")
                            else:
                                # Show basic data table if no creatives
                                st.subheader("📊 Campaign Data")
                                display_columns = ['campaign_name', 'impressions', 'clicks', 'spend', 'ctr']
                                available_columns = [col for col in display_columns if col in paid_data.columns]
                                if available_columns:
                                    st.dataframe(paid_data[available_columns])
                                else:
                                    st.dataframe(paid_data)
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
                                            st.subheader("📸 Instagram Media Insights")

                                            # Show metrics breakdown
                                            metrics_available = ig_data['metric'].unique().tolist()
                                            st.info(f"Available metrics: {', '.join(metrics_available)}")

                                            # Build unique media list with readable labels
                                            ig_unique = ig_data[['media_id', 'timestamp', 'caption', 'media_url', 'permalink', 'thumbnail_url']].drop_duplicates(subset=['media_id'])
                                            
                                            if not ig_unique.empty:
                                                # Create readable labels for post selection
                                                labels = {}
                                                for _, row in ig_unique.iterrows():
                                                    date_part = row['timestamp'].split('T')[0] if row['timestamp'] else 'Unknown date'
                                                    caption_part = row['caption'][:50] + "..." if row['caption'] and len(row['caption']) > 50 else row['caption'] or 'No caption'
                                                    labels[row['media_id']] = f"{date_part}: {caption_part}"

                                                # Media selector with readable labels
                                                selected_media = st.selectbox(
                                                    "Select post to inspect:", 
                                                    options=list(labels.keys()), 
                                                    format_func=lambda mid: labels[mid]
                                                )
                                                
                                                # Show selected post details
                                                sel_row = ig_unique[ig_unique['media_id'] == selected_media].iloc[0]
                                                
                                                col_post_img, col_post_details = st.columns([1, 2])
                                                
                                                with col_post_img:
                                                    img_url = sel_row.get('media_url') or sel_row.get('thumbnail_url')
                                                    if img_url and pd.notna(img_url):
                                                        try:
                                                            st.image(img_url, width=300, caption="Instagram Post")
                                                        except Exception as e:
                                                            st.warning(f"Could not load image: {e}")
                                                    else:
                                                        st.info("No preview image available")
                                                
                                                with col_post_details:
                                                    if pd.notna(sel_row.get('permalink')):
                                                        st.markdown(f"🔗 [View on Instagram]({sel_row['permalink']})")
                                                    
                                                    if pd.notna(sel_row.get('caption')):
                                                        st.markdown(f"**Caption:** {sel_row['caption']}")
                                                    
                                                    # Show metrics for this post
                                                    sel_metrics = ig_data[ig_data['media_id'] == selected_media][['metric', 'value']]
                                                    if not sel_metrics.empty:
                                                        st.subheader("📊 Post Metrics")
                                                        
                                                        # Display metrics in columns for better readability
                                                        metrics_cols = st.columns(min(len(sel_metrics), 4))
                                                        for idx, (_, metric_row) in enumerate(sel_metrics.iterrows()):
                                                            with metrics_cols[idx % 4]:
                                                                st.metric(
                                                                    metric_row['metric'].replace('_', ' ').title(),
                                                                    f"{int(metric_row['value']):,}" if metric_row['value'] else "0"
                                                                )
                                                        
                                                        # Bar chart of metrics for this post
                                                        fig = go.Figure(data=[
                                                            go.Bar(
                                                                x=sel_metrics['metric'],
                                                                y=sel_metrics['value'],
                                                                text=[f"{int(x):,}" for x in sel_metrics['value']],
                                                                textposition='auto',
                                                            )
                                                        ])
                                                        fig.update_layout(
                                                            title="Post Performance Metrics",
                                                            xaxis_title="Metric",
                                                            yaxis_title="Value",
                                                            xaxis_tickangle=-45,
                                                            height=400
                                                        )
                                                        st.plotly_chart(fig, use_container_width=True)

                                                st.markdown("---")
                                                
                                                # Aggregate metrics over time with improved readability
                                                st.subheader("📈 Performance Over Time")
                                                ig_data['date'] = pd.to_datetime(ig_data['timestamp']).dt.date
                                                df_pivot = ig_data.pivot_table(
                                                    index='date', 
                                                    columns='metric', 
                                                    values='value', 
                                                    aggfunc='sum'
                                                ).reset_index()
                                                
                                                if not df_pivot.empty:
                                                    fig_time = go.Figure()
                                                    
                                                    if 'impressions' in df_pivot.columns:
                                                        fig_time.add_trace(go.Scatter(
                                                            x=df_pivot['date'],
                                                            y=df_pivot['impressions'],
                                                            mode='lines+markers',
                                                            name='Impressions',
                                                            line=dict(width=3)
                                                        ))
                                                    
                                                    if 'reach' in df_pivot.columns:
                                                        fig_time.add_trace(go.Scatter(
                                                            x=df_pivot['date'],
                                                            y=df_pivot['reach'],
                                                            mode='lines+markers',
                                                            name='Reach',
                                                            line=dict(width=3)
                                                        ))
                                                    
                                                    fig_time.update_layout(
                                                        title="Instagram Performance Trends",
                                                        xaxis_title="Date",
                                                        yaxis_title="Value",
                                                        height=400,
                                                        legend=dict(x=0, y=1)
                                                    )
                                                    st.plotly_chart(fig_time, use_container_width=True)
                                            else:
                                                st.warning("No Instagram media found for the selected period")
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