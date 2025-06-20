"""
Streamlit dashboard with improved UI layout, creative previews, and no column nesting.
Official docs: https://docs.streamlit.io/
"""
import os
import logging
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="AI Campaign Optimizer",
    page_icon="🎯",
    layout="wide"
)

def check_environment():
    """Check required environment variables and Facebook SDK imports."""
    required_env = ["META_ACCESS_TOKEN", "AD_ACCOUNT_ID"]
    missing_env = [k for k in required_env if not os.getenv(k)]

    optional_env = ["META_APP_ID", "META_APP_SECRET", "PAGE_ID", "PAGE_ACCESS_TOKEN", "IG_USER_ID"]
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

@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_get_paid_with_creatives(date_preset):
    """Cached version of paid campaign fetch with creatives."""
    try:
        from fetch_paid import get_campaign_performance_with_creatives
        return get_campaign_performance_with_creatives(date_preset=date_preset)
    except Exception as e:
        logger.error(f"Error in cached_get_paid_with_creatives: {e}", exc_info=True)
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_fetch_ig_insights(ig_user_id, since, until, metrics):
    """Cached version of Instagram insights fetch."""
    try:
        from fetch_organic import fetch_ig_media_insights
        return fetch_ig_media_insights(ig_user_id, since=since, until=until, metrics=metrics)
    except Exception as e:
        logger.error(f"Error in cached_fetch_ig_insights: {e}", exc_info=True)
        return pd.DataFrame()

def show_paid_section():
    """Display paid campaigns section with creative previews."""
    st.header("🎯 Paid Campaigns with Creative Previews")

    # Controls in sidebar
    with st.sidebar:
        st.subheader("Paid Campaign Settings")
        date_preset = st.selectbox(
            "Select time range:",
            ["yesterday", "last_7d", "last_30d", "this_month", "last_month"],
            index=1,
            key="paid_preset"
        )

        fetch_paid = st.button("Fetch Paid Data", key="fetch_paid_btn")

    if fetch_paid:
        with st.spinner("Fetching campaign data with creatives..."):
            try:
                df_paid = cached_get_paid_with_creatives(date_preset)

                if df_paid.empty:
                    st.warning(f"No paid campaign data available for {date_preset}")
                    return

                st.success(f"✅ Fetched {len(df_paid)} campaign records with creatives")

                # Summary metrics at top level
                total_spend = df_paid['spend'].sum() if 'spend' in df_paid.columns else 0
                total_impr = df_paid['impressions'].sum() if 'impressions' in df_paid.columns else 0
                total_clicks = df_paid['clicks'].sum() if 'clicks' in df_paid.columns else 0
                avg_ctr = (total_clicks / total_impr * 100) if total_impr > 0 else 0

                # Single level columns for metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Spend", f"${total_spend:.2f}")
                with col2:
                    st.metric("Total Impressions", f"{int(total_impr):,}")
                with col3:
                    st.metric("Total Clicks", f"{int(total_clicks):,}")
                with col4:
                    st.metric("Avg CTR", f"{avg_ctr:.2f}%")

                # Campaign spend chart
                if 'spend' in df_paid.columns and 'campaign_name' in df_paid.columns:
                    campaign_sums = df_paid.groupby('campaign_name')['spend'].sum().reset_index()
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

                st.markdown("---")

                # Creative previews section
                st.subheader("🎨 Ad Creative Previews")

                # Group by campaign for better organization
                campaigns = df_paid['campaign_name'].unique() if 'campaign_name' in df_paid.columns else []

                for campaign_name in campaigns:
                    if pd.notna(campaign_name):
                        campaign_data = df_paid[df_paid['campaign_name'] == campaign_name]

                        with st.expander(f"📊 {campaign_name}", expanded=True):
                            for _, row in campaign_data.iterrows():
                                if pd.notna(row.get('ad_name')):
                                    # Single level columns for each ad
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

                                        # Performance metrics - single level columns
                                        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                                        with metrics_col1:
                                            st.metric("Impressions", f"{int(row.get('impressions', 0)):,}")
                                        with metrics_col2:
                                            st.metric("Clicks", f"{int(row.get('clicks', 0)):,}")
                                        with metrics_col3:
                                            st.metric("Spend", f"${row.get('spend', 0):.2f}")
                                        with metrics_col4:
                                            ctr = row.get('ctr', 0)
                                            st.metric("CTR", f"{ctr:.2f}%" if ctr else "0%")

                                        if pd.notna(row.get('creative_object_url')):
                                            st.markdown(f"🔗 [View Ad Destination]({row['creative_object_url']})")

                                    st.markdown("---")

            except Exception as e:
                st.error(f"Error fetching paid data: {e}")
                logger.error(f"❌ Error in show_paid_section: {e}", exc_info=True)
    else:
        st.info("Click 'Fetch Paid Data' in the sidebar to load campaign data with creative previews")

def show_instagram_section():
    """Display Instagram insights section with post previews."""
    st.header("📸 Instagram Media Insights with Previews")

    ig_user_id = os.getenv("IG_USER_ID")
    if not ig_user_id:
        st.error("❌ IG_USER_ID not set in environment variables")
        st.info("Add IG_USER_ID to Replit Secrets to enable Instagram insights")
        return

    # Controls in sidebar
    with st.sidebar:
        st.subheader("Instagram Settings")
        since = st.date_input("Since", value=(date.today() - timedelta(days=7)), key="ig_since")
        until = st.date_input("Until", value=(date.today() - timedelta(days=1)), key="ig_until")

        if since > until:
            st.error("❌ Start date must be before end date")
            return

        fetch_ig = st.button("Fetch Instagram Data", key="fetch_ig_btn")

    if fetch_ig:
        with st.spinner("Fetching Instagram insights..."):
            try:
                since_str = since.strftime("%Y-%m-%d")
                until_str = until.strftime("%Y-%m-%d")

                ig_data = cached_fetch_ig_insights(
                    ig_user_id, 
                    since_str, 
                    until_str, 
                    ["impressions", "reach", "total_interactions"]
                )

                if ig_data.empty:
                    st.warning(f"No Instagram data for {since_str} to {until_str}")
                    return

                st.success(f"✅ Fetched {len(ig_data)} Instagram insights records")
                st.info(f"Available metrics: {', '.join(ig_data['metric'].unique())}")

                # Get unique posts
                ig_unique = ig_data[['media_id', 'timestamp', 'caption', 'media_url', 'permalink', 'thumbnail_url']].drop_duplicates(subset=['media_id'])

                if ig_unique.empty:
                    st.warning("No Instagram posts found")
                    return

                # Create readable labels for post selection
                labels = {}
                for _, row in ig_unique.iterrows():
                    date_part = row['timestamp'].split('T')[0] if row['timestamp'] else 'Unknown date'
                    caption_part = row['caption'][:50] + "..." if row['caption'] and len(row['caption']) > 50 else row['caption'] or 'No caption'
                    labels[row['media_id']] = f"{date_part}: {caption_part}"

                # Post selector
                selected_media = st.selectbox(
                    "Select post to inspect:", 
                    options=list(labels.keys()), 
                    format_func=lambda mid: labels[mid],
                    key="ig_post_selector"
                )

                # Show selected post details - single level columns
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

                    if pd.notna(sel_row.get('permalink')):
                        st.markdown(f"🔗 [View on Instagram]({sel_row['permalink']})")

                with col_post_details:
                    if pd.notna(sel_row.get('caption')):
                        st.markdown(f"**Caption:** {sel_row['caption']}")

                    # Show metrics for this post
                    sel_metrics = ig_data[ig_data['media_id'] == selected_media][['metric', 'value']]
                    if not sel_metrics.empty:
                        st.subheader("📊 Post Metrics")

                        # Display metrics in single level columns
                        num_metrics = len(sel_metrics)
                        if num_metrics <= 4:
                            metric_cols = st.columns(num_metrics)
                            for idx, (_, metric_row) in enumerate(sel_metrics.iterrows()):
                                with metric_cols[idx]:
                                    st.metric(
                                        metric_row['metric'].replace('_', ' ').title(),
                                        f"{int(metric_row['value']):,}" if metric_row['value'] else "0"
                                    )
                        else:
                            # Too many metrics, use DataFrame
                            st.dataframe(sel_metrics.set_index('metric'))

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

                # Aggregate metrics over time
                st.subheader("📈 Instagram Performance Over Time")
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

            except Exception as e:
                st.error(f"Error fetching Instagram data: {e}")
                logger.error(f"❌ Error in show_instagram_section: {e}", exc_info=True)
    else:
        st.info("Click 'Fetch Instagram Data' in the sidebar to load Instagram insights with post previews")

def main():
    """Main dashboard function."""
    logger.info("🚀 Starting Streamlit app")

    # Environment check
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
        from fb_client import fb_client
        from fetch_paid import get_campaign_performance_with_creatives
        from fetch_organic import fetch_ig_media_insights
    except ImportError as e:
        st.error(f"❌ Failed to import modules: {e}")
        logger.error(f"❌ Module import error: {e}", exc_info=True)
        st.stop()

    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")

        # API connection status
        if fb_client.is_initialized():
            st.success("✅ Facebook API Connected")

            if st.button("Test Connection"):
                with st.spinner("Testing connection..."):
                    test_result = fb_client.test_connection()
                    if test_result["success"]:
                        st.success("✅ Connection successful!")
                        st.json(test_result)
                    else:
                        st.error(f"❌ Connection failed: {test_result['error']}")
        else:
            st.error("❌ Facebook API Not Connected")
            st.info("Check your Meta credentials in Replit Secrets")

        # Environment status
        st.subheader("📊 Environment Status")
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

    # Main content tabs
    tab1, tab2 = st.tabs(["🎯 Paid Campaigns", "📸 Instagram Insights"])

    with tab1:
        show_paid_section()

    with tab2:
        show_instagram_section()

    # Footer
    st.markdown("---")
    st.markdown("🚀 AI-Powered Campaign Optimizer - Built with ❤️ on Replit")
    st.markdown("**Official Documentation:**")
    st.markdown("- [Facebook Business SDK](https://developers.facebook.com/docs/business-sdk/)")
    st.markdown("- [Marketing API Insights](https://developers.facebook.com/docs/marketing-api/insights/)")
    st.markdown("- [Instagram API](https://developers.facebook.com/docs/instagram-api/)")
    st.markdown("- [Streamlit Documentation](https://docs.streamlit.io/)")

if __name__ == "__main__":
    main()