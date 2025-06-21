"""
AI-Powered Social Campaign Optimizer Dashboard
Enhanced with comprehensive Instagram insights, metadata-driven metrics, and AI commentary
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Any
import requests
import re

# Import our modules
from fetch_organic import (
    fetch_ig_media_insights, 
    get_ig_follower_count, 
    fetch_ig_user_insights,
    compute_instagram_kpis,
    validate_organic_environment
)

# Import paid insights functions with error handling
try:
    from fetch_paid import get_campaign_performance_with_creatives, get_paid_insights
    PAID_INSIGHTS_AVAILABLE = True
    logger.info("✅ Paid insights functions imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import paid insights functions: {e}")
    get_campaign_performance_with_creatives = None
    get_paid_insights = None
    PAID_INSIGHTS_AVAILABLE = False

# Configure OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    openai.api_key = openai_api_key
    logger.info("OpenAI API key configured")
else:
    logger.warning("OpenAI API key not found")

# Set page config
st.set_page_config(
    page_title="AI-Powered Social Campaign Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import openai
from config import config

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data(ttl=600)
def cached_fetch_ig_media_insights(ig_user_id: str, since: str = None, until: str = None) -> pd.DataFrame:
    """Cached version of fetch_ig_media_insights with 10-minute TTL"""
    return fetch_ig_media_insights(ig_user_id, since, until)

@st.cache_data(ttl=1200)
def cached_get_ig_follower_count(ig_user_id: str) -> Optional[int]:
    """Cached version of get_ig_follower_count with 20-minute TTL"""
    return get_ig_follower_count(ig_user_id)

@st.cache_data(ttl=300)
def cached_openai_commentary(media_id: str, context: str) -> str:
    """Cached OpenAI commentary to avoid repeated API calls"""
    if not openai_api_key:
        return "⚠️ OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert Instagram marketing analyst. Provide actionable insights and recommendations based on performance data. Focus on 2025 trends and best practices."
                },
                {
                    "role": "user", 
                    "content": f"Analyze this Instagram post performance and provide 2-3 specific, actionable recommendations for 2025:\n\n{context}"
                }
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return f"⚠️ AI commentary failed: {str(e)}"

def analyze_caption(caption: str) -> Dict[str, Any]:
    """Analyze caption for word count, hashtags, emojis"""
    if not caption:
        return {"word_count": 0, "hashtag_count": 0, "emoji_count": 0, "has_cta": False}

    # Word count
    words = len(caption.split())

    # Hashtag count
    hashtags = len(re.findall(r'#\w+', caption))

    # Emoji count (basic)
    emojis = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', caption))

    # CTA detection
    cta_words = ['link', 'bio', 'swipe', 'comment', 'share', 'save', 'follow', 'click', 'tap', 'visit', 'shop', 'buy']
    has_cta = any(word.lower() in caption.lower() for word in cta_words)

    return {
        "word_count": words,
        "hashtag_count": hashtags,
        "emoji_count": emojis,
        "has_cta": has_cta
    }

def compute_posting_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze posting patterns and performance"""
    if df.empty:
        return {}

    # Convert timestamp to datetime
    df_copy = df.copy()
    df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'])
    df_copy['weekday'] = df_copy['datetime'].dt.day_name()
    df_copy['hour'] = df_copy['datetime'].dt.hour

    # Get engagement metrics per post
    engagement_df = df_copy[df_copy['metric'] == 'total_interactions'].copy()
    if engagement_df.empty:
        return {}

    # Best performing weekdays
    weekday_performance = engagement_df.groupby('weekday')['value'].mean().sort_values(ascending=False)

    # Best performing hours
    hour_performance = engagement_df.groupby('hour')['value'].mean().sort_values(ascending=False)

    return {
        'best_weekday': weekday_performance.index[0] if not weekday_performance.empty else "N/A",
        'best_hour': hour_performance.index[0] if not hour_performance.empty else "N/A",
        'avg_engagement': engagement_df['value'].mean(),
        'total_posts': len(engagement_df)
    }

def show_instagram_insights():
    """Main Instagram insights dashboard section"""
    st.header("📸 Instagram Media Insights")

    # Environment validation
    validation = validate_organic_environment()

    if not validation['instagram_insights_enabled']:
        if not validation['page_token_available']:
            st.error("❌ Missing PAGE_ACCESS_TOKEN. Please set this environment variable.")
            st.stop()

        if not validation['ig_user_id_available']:
            st.error("❌ Missing IG_USER_ID. Please set this environment variable.")
            st.stop()

    # Get environment variables
    ig_user_id = os.getenv('IG_USER_ID')

    # Sidebar controls
    with st.sidebar:
        st.subheader("📅 Date Range")

        # Date range selector
        today = date.today()
        default_start = today - timedelta(days=7)

        since_date = st.date_input(
            "Since",
            value=default_start,
            max_value=today
        )

        until_date = st.date_input(
            "Until", 
            value=today,
            max_value=today
        )

        # Convert to strings
        since_str = since_date.strftime("%Y-%m-%d")
        until_str = until_date.strftime("%Y-%m-%d")

        # Fetch button
        fetch_data = st.button("🔄 Fetch Instagram Data", type="primary")

        # Limit posts option
        limit_posts = st.checkbox("Limit to recent posts", value=True)
        if limit_posts:
            max_posts = st.slider("Max posts to analyze", 5, 50, 20)

    # Main content area
    if fetch_data or 'ig_data' not in st.session_state:
        with st.spinner("Fetching Instagram insights..."):
            try:
                # Fetch data
                df = cached_fetch_ig_media_insights(ig_user_id, since_str, until_str)
                follower_count = cached_get_ig_follower_count(ig_user_id)

                # Store in session state
                st.session_state.ig_data = df
                st.session_state.follower_count = follower_count

                if df.empty:
                    st.warning("⚠️ No Instagram insights returned. Check your token permissions, Instagram Business account linkage, and date range.")
                    return

                # Limit posts if requested
                if limit_posts and not df.empty:
                    unique_media = df['media_id'].unique()[:max_posts]
                    df = df[df['media_id'].isin(unique_media)]
                    st.session_state.ig_data = df

                st.success(f"✅ Fetched {len(df)} records from {len(df['media_id'].unique()) if not df.empty else 0} posts")

            except Exception as e:
                st.error(f"❌ Error fetching Instagram data: {str(e)}")
                logger.error(f"Instagram fetch error: {e}", exc_info=True)
                return

    # Use cached data
    df = st.session_state.get('ig_data', pd.DataFrame())
    follower_count = st.session_state.get('follower_count', 0)

    if df.empty:
        st.info("👆 Click 'Fetch Instagram Data' to load your insights")
        return

    # Show available metrics
    available_metrics = sorted(df['metric'].unique())
    st.info(f"📊 **Available metrics:** {', '.join(available_metrics)}")

    # Create two columns for layout
    col1, col2 = st.columns([1, 2])

    with col1:
        # Post selector
        st.subheader("📱 Select Post")

        post_options = {}
        for _, row in df[['media_id', 'timestamp', 'caption']].drop_duplicates().iterrows():
            date_str = pd.to_datetime(row['timestamp']).strftime("%Y-%m-%d")
            caption_preview = (row['caption'][:50] + "...") if len(row['caption']) > 50 else row['caption']
            label = f"{date_str}: {caption_preview}"
            post_options[label] = row['media_id']

        if not post_options:
            st.warning("No posts available to select")
            return

        selected_label = st.selectbox(
            "Choose a post to analyze:",
            options=list(post_options.keys()),
            key="post_selector"
        )

        selected_media_id = post_options[selected_label]

        # Get selected post data
        post_df = df[df['media_id'] == selected_media_id]
        post_info = post_df.iloc[0]

        # Media preview
        st.subheader("🎥 Media Preview")

        media_url = post_info['media_url']
        thumbnail_url = post_info['thumbnail_url']
        media_type = post_info['media_type']
        permalink = post_info['permalink']

        try:
            if media_type == 'VIDEO' or post_info['media_product_type'] == 'REEL':
                if media_url:
                    st.video(media_url)
                elif thumbnail_url:
                    st.image(thumbnail_url, caption="Video thumbnail")
            else:
                if media_url:
                    st.image(media_url, caption="Post image")
                elif thumbnail_url:
                    st.image(thumbnail_url, caption="Post thumbnail")
        except Exception as e:
            st.warning(f"Could not load media: {str(e)}")

        if permalink:
            st.markdown(f"🔗 [View on Instagram]({permalink})")

    with col2:
        # Post metrics
        st.subheader("📊 Post Metrics")

        # Create metrics pivot
        metrics_map = dict(zip(post_df['metric'], post_df['value']))

        # Display key metrics in columns
        metric_cols = st.columns(4)

        with metric_cols[0]:
            reach = metrics_map.get('reach', 0)
            st.metric("Reach", f"{reach:,}")

        with metric_cols[1]:
            interactions = metrics_map.get('total_interactions', 0)
            st.metric("Interactions", f"{interactions:,}")

        with metric_cols[2]:
            if follower_count and follower_count > 0:
                engagement_rate = (interactions / follower_count) * 100
                st.metric("Engagement Rate", f"{engagement_rate:.2f}%")
            else:
                st.metric("Engagement Rate", "N/A")

        with metric_cols[3]:
            saves = metrics_map.get('saved', 0)
            st.metric("Saves", f"{saves:,}")

        # Detailed metrics table
        st.subheader("📋 All Metrics")
        metrics_df = post_df[['metric', 'value']].sort_values('value', ascending=False)
        st.dataframe(metrics_df, use_container_width=True)

    # KPIs and Analysis
    with st.expander("🎯 Computed KPIs & Engagement Analysis", expanded=True):
        kpis = compute_instagram_kpis(post_df, follower_count)

        if kpis:
            kpi_cols = st.columns(3)

            with kpi_cols[0]:
                if 'engagement_rate_by_reach' in kpis:
                    st.metric("Engagement Rate (Reach)", f"{kpis['engagement_rate_by_reach']:.2f}%")

                if 'save_rate' in kpis:
                    st.metric("Save Rate", f"{kpis['save_rate']:.2f}%")

            with kpi_cols[1]:
                if 'profile_visit_rate' in kpis:
                    st.metric("Profile Visit Rate", f"{kpis['profile_visit_rate']:.2f}%")

                if 'follow_rate' in kpis:
                    st.metric("Follow Rate", f"{kpis['follow_rate']:.2f}%")

            with kpi_cols[2]:
                if follower_count:
                    st.metric("Follower Count", f"{follower_count:,}")

                if 'avg_reels_watch_time' in kpis and kpis['avg_reels_watch_time'] > 0:
                    st.metric("Avg Watch Time", f"{kpis['avg_reels_watch_time']:.1f}s")

        # Historical comparison
        st.subheader("📈 Historical Comparison")

        posting_insights = compute_posting_insights(df)
        if posting_insights:
            comp_cols = st.columns(2)

            with comp_cols[0]:
                current_engagement = interactions
                avg_engagement = posting_insights.get('avg_engagement', 0)

                if avg_engagement > 0:
                    performance_vs_avg = ((current_engagement - avg_engagement) / avg_engagement) * 100

                    if performance_vs_avg > 10:
                        st.success(f"🎉 This post performs {performance_vs_avg:.1f}% above your average!")
                    elif performance_vs_avg > 0:
                        st.info(f"📊 This post performs {performance_vs_avg:.1f}% above average")
                    else:
                        st.warning(f"📉 This post performs {abs(performance_vs_avg):.1f}% below average")
                else:
                    st.info("Not enough historical data for comparison")

            with comp_cols[1]:
                if posting_insights.get('best_weekday'):
                    st.info(f"🗓️ **Best posting day:** {posting_insights['best_weekday']}")

                if posting_insights.get('best_hour'):
                    st.info(f"🕐 **Best posting hour:** {posting_insights['best_hour']}:00")

    # Caption Analysis
    with st.expander("✍️ Caption Analysis"):
        caption = post_info['caption']
        caption_analysis = analyze_caption(caption)

        cap_cols = st.columns(4)

        with cap_cols[0]:
            st.metric("Word Count", caption_analysis['word_count'])

        with cap_cols[1]:
            st.metric("Hashtags", caption_analysis['hashtag_count'])

        with cap_cols[2]:
            st.metric("Emojis", caption_analysis['emoji_count'])

        with cap_cols[3]:
            cta_status = "✅ Yes" if caption_analysis['has_cta'] else "❌ No"
            st.metric("Has CTA", cta_status)

        if caption:
            with st.expander("Full Caption"):
                st.text(caption)

    # AI Commentary
    with st.expander("🤖 AI Commentary & Recommendations"):
        if openai_api_key:
            # Prepare context for AI
            # Calculate engagement rate
            if follower_count and follower_count > 0:
                engagement_rate = ((metrics_map.get('total_interactions', 0) / follower_count) * 100)
                engagement_rate_str = f"{engagement_rate:.2f}%"
            else:
                engagement_rate_str = "N/A"

            context = f"""
            Post Date: {pd.to_datetime(post_info['timestamp']).strftime('%Y-%m-%d')}
            Media Type: {post_info['media_type']} / {post_info['media_product_type']}
            Caption: {caption[:200]}...

            Key Metrics:
            - Reach: {metrics_map.get('reach', 0):,}
            - Total Interactions: {metrics_map.get('total_interactions', 0):,}
            - Comments: {metrics_map.get('comments', 0):,}
            - Shares: {metrics_map.get('shares', 0):,}
            - Saves: {metrics_map.get('saved', 0):,}
            - Profile Visits: {metrics_map.get('profile_visits', 0):,}
            - Follower Count: {follower_count:,}

            Engagement Rate: {engagement_rate_str}
            """

            ai_commentary = cached_openai_commentary(selected_media_id, context)
            st.write(ai_commentary)
        else:
            st.warning("⚠️ OpenAI API key not configured. Set OPENAI_API_KEY environment variable to enable AI commentary.")

    # Trends Over Time
    with st.expander("📈 Instagram Trends Over Time"):
        if len(df['media_id'].unique()) > 1:
            # Metric selector
            selected_metrics = st.multiselect(
                "Select metrics to plot:",
                options=available_metrics,
                default=['reach', 'total_interactions'] if 'reach' in available_metrics else available_metrics[:2]
            )

            if selected_metrics:
                # Pivot data by date
                plot_df = df[df['metric'].isin(selected_metrics)].copy()
                plot_df['date'] = pd.to_datetime(plot_df['timestamp']).dt.date

                # Create pivot table
                pivot_df = plot_df.pivot_table(
                    index='date', 
                    columns='metric', 
                    values='value', 
                    aggfunc='sum'
                ).fillna(0)

                # Create matplotlib figure
                fig, ax = plt.subplots(figsize=(12, 6))

                for metric in selected_metrics:
                    if metric in pivot_df.columns:
                        ax.plot(pivot_df.index, pivot_df[metric], marker='o', label=metric, linewidth=2)

                ax.set_title('Instagram Metrics Over Time', fontsize=16, fontweight='bold')
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Value', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Rotate x-axis labels
                plt.xticks(rotation=45)
                plt.tight_layout()

                st.pyplot(fig)
            else:
                st.info("Select metrics to display the trend chart")
        else:
            st.info("Need more than one post to show trends over time")

def show_paid_campaign_insights():
    """Paid campaign insights section"""
    st.header("💰 Paid Campaign Insights")

    if not PAID_INSIGHTS_AVAILABLE:
        st.error("❌ Paid insights functionality not available. Please check your fetch_paid.py configuration and ensure all dependencies are installed.")
        st.info("Required: facebook-business SDK and properly configured fb_client.py")
        return

    # Sidebar controls for paid insights
    with st.sidebar:
        st.subheader("📅 Paid Campaign Settings")
        
        date_preset = st.selectbox(
            "Date Range",
            options=["yesterday", "last_7d", "last_30d", "this_month", "last_month"],
            index=1  # Default to last_7d
        )
        
        include_creatives = st.checkbox("Include Creative Previews", value=True)

    with st.spinner("Loading paid campaign data..."):
        try:
            # Use the correct function with proper parameters
            paid_data = get_campaign_performance_with_creatives(
                date_preset=date_preset, 
                include_creatives=include_creatives
            )

            if paid_data.empty:
                st.warning("No paid campaign data available. Check your Meta Ads account connection and ensure AD_ACCOUNT_ID is set.")
                return

            # Display summary metrics
            if len(paid_data) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_spend = paid_data['spend'].astype(float).sum()
                    st.metric("Total Spend", f"${total_spend:,.2f}")
                
                with col2:
                    total_impressions = paid_data['impressions'].astype(int).sum()
                    st.metric("Total Impressions", f"{total_impressions:,}")
                
                with col3:
                    total_clicks = paid_data['clicks'].astype(int).sum()
                    st.metric("Total Clicks", f"{total_clicks:,}")
                
                with col4:
                    avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
                    st.metric("Average CTR", f"{avg_ctr:.2f}%")

            # Show campaign data table
            st.subheader("📊 Campaign Performance Data")
            
            # Format numeric columns for better display
            display_df = paid_data.copy()
            if 'spend' in display_df.columns:
                display_df['spend'] = display_df['spend'].astype(float).round(2)
            if 'ctr' in display_df.columns:
                display_df['ctr'] = display_df['ctr'].astype(float).round(3)
            if 'cpc' in display_df.columns:
                display_df['cpc'] = display_df['cpc'].astype(float).round(3)
            
            st.dataframe(display_df, use_container_width=True)

            # Show creative previews if available
            if include_creatives and 'creative_image_url' in paid_data.columns:
                st.subheader("🎨 Creative Previews")
                
                # Filter rows with creative URLs
                creative_rows = paid_data[paid_data['creative_image_url'].notna()].head(5)
                
                if not creative_rows.empty:
                    for _, row in creative_rows.iterrows():
                        with st.expander(f"Creative: {row.get('creative_name', 'Unnamed')}"):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                if pd.notna(row.get('creative_image_url')):
                                    try:
                                        st.image(row['creative_image_url'], caption="Creative Preview")
                                    except:
                                        st.text("Preview not available")
                            
                            with col2:
                                st.write(f"**Campaign:** {row.get('campaign_name', 'N/A')}")
                                st.write(f"**Ad:** {row.get('ad_name', 'N/A')}")
                                if pd.notna(row.get('creative_body')):
                                    st.write(f"**Body:** {row.get('creative_body')}")
                                if pd.notna(row.get('creative_title')):
                                    st.write(f"**Title:** {row.get('creative_title')}")
                else:
                    st.info("No creative previews available for current campaigns.")

        except Exception as e:
            st.error(f"Error loading paid campaign data: {str(e)}")
            logger.error(f"Paid campaign error: {e}", exc_info=True)

def main():
    """Main dashboard function"""
    st.title("🚀 AI-Powered Social Campaign Optimizer")
    st.markdown("*Comprehensive Instagram insights with AI-driven recommendations for 2025*")

    # Create tabs for different sections
    tab1, tab2 = st.tabs(["📸 Instagram Insights", "💰 Paid Campaigns"])

    with tab1:
        show_instagram_insights()

    with tab2:
        show_paid_campaign_insights()

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit • Powered by Meta Graph API & OpenAI*")

if __name__ == "__main__":
    main()