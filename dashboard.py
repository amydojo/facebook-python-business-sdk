
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

# Configure logging FIRST before any other imports that might use logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import OpenAI with error handling
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not available - AI features disabled")

# Import our modules
from fetch_organic import (
    fetch_ig_media_insights, 
    get_ig_follower_count, 
    fetch_ig_user_insights,
    compute_instagram_kpis,
    validate_organic_environment
)

# Import optimized API helpers
from api_helpers import get_api_stats

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
if openai_api_key and OPENAI_AVAILABLE:
    try:
        # Set the API key for OpenAI
        openai.api_key = openai_api_key
        logger.info("✅ OpenAI API key configured")
    except Exception as e:
        logger.error(f"Failed to configure OpenAI: {e}")
        OPENAI_AVAILABLE = False
else:
    if not OPENAI_AVAILABLE:
        logger.warning("OpenAI package not installed")
    else:
        logger.warning("OPENAI_API_KEY not set: AI commentary disabled")

# Set page config
st.set_page_config(
    page_title="AI-Powered Social Campaign Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

from config import config

@st.cache_data(ttl=600)
def cached_fetch_ig_media_insights(ig_user_id: str, since: str = None, until: str = None) -> pd.DataFrame:
    """Cached version of fetch_ig_media_insights with 10-minute TTL"""
    try:
        return fetch_ig_media_insights(ig_user_id, since, until)
    except Exception as e:
        logger.error(f"Error in cached_fetch_ig_media_insights: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1200)
def cached_get_ig_follower_count(ig_user_id: str) -> Optional[int]:
    """Cached version of get_ig_follower_count with 20-minute TTL"""
    try:
        return get_ig_follower_count(ig_user_id)
    except Exception as e:
        logger.error(f"Error in cached_get_ig_follower_count: {e}")
        return None

@st.cache_data(ttl=300)
def cached_openai_commentary(media_id: str, context: str) -> str:
    """Cached OpenAI commentary to avoid repeated API calls"""
    if not openai_api_key or not OPENAI_AVAILABLE:
        return "⚠️ OpenAI not available. Please install openai package and set OPENAI_API_KEY environment variable."

    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)
        
        response = client.chat.completions.create(
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
    try:
        df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'])
        df_copy['weekday'] = df_copy['datetime'].dt.day_name()
        df_copy['hour'] = df_copy['datetime'].dt.hour
    except Exception as e:
        logger.warning(f"Error parsing timestamps: {e}")
        return {}

    # Get engagement metrics per post - use available metrics
    available_metrics = df_copy['metric'].unique()
    
    # Try to find engagement metrics
    engagement_metric = None
    for metric in ['total_interactions', 'likes', 'reach']:
        if metric in available_metrics:
            engagement_metric = metric
            break
    
    if not engagement_metric:
        return {}

    engagement_df = df_copy[df_copy['metric'] == engagement_metric].copy()
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
                st.session_state.last_fetch_time = datetime.now()

                if df.empty:
                    st.warning("⚠️ No Instagram insights returned. This could be due to:")
                    st.write("• **Rate limiting** - Meta limits API calls per hour")
                    st.write("• **Token permissions** - Check pages_read_engagement & instagram_manage_insights scopes")
                    st.write("• **Instagram Business account** - Ensure account is properly linked")
                    st.write("• **Date range** - No content published in selected period")
                    
                    # Show API usage stats
                    api_stats = get_api_stats()
                    if api_stats['total_calls'] > 150:
                        st.error(f"🚫 High API usage: {api_stats['total_calls']} calls in {api_stats['session_duration_minutes']:.1f} minutes")
                        st.info("**Recommendation:** Wait 1 hour before making more requests, or use cached data")
                    
                    # Show troubleshooting tips
                    with st.expander("🔧 Troubleshooting Tips"):
                        st.write("1. **Check token scopes** in Meta Developer console")
                        st.write("2. **Verify Instagram Business account** is connected to Facebook Page")
                        st.write("3. **Try a different date range** with known content")
                        st.write("4. **Check API limits** - Meta allows ~200 calls per hour")
                        if 'last_fetch_time' in st.session_state:
                            last_fetch = st.session_state.last_fetch_time
                            st.write(f"5. **Last successful fetch:** {last_fetch.strftime('%Y-%m-%d %H:%M:%S')}")
                    return

                # Limit posts if requested
                if limit_posts and not df.empty:
                    unique_media = df['media_id'].unique()[:max_posts]
                    df = df[df['media_id'].isin(unique_media)]
                    st.session_state.ig_data = df

                # Show success with API stats
                api_stats = get_api_stats()
                st.success(f"✅ Fetched {len(df)} records from {len(df['media_id'].unique()) if not df.empty else 0} posts")
                
                # Show rate limit warning if approaching limits
                if api_stats['calls_per_minute'] > 2.5:  # Over 150 calls per hour
                    st.warning(f"⚠️ High API usage: {api_stats['calls_per_minute']:.1f} calls/min. Consider using cached data.")

            except Exception as e:
                error_msg = str(e).lower()
                if 'rate limit' in error_msg or 'user request limit' in error_msg:
                    st.error("🚫 Rate limit reached. Please wait a few minutes before refreshing.")
                    if 'last_fetch_time' in st.session_state:
                        last_fetch = st.session_state.last_fetch_time
                        st.info(f"Last successful fetch: {last_fetch.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
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
            try:
                date_str = pd.to_datetime(row['timestamp']).strftime("%Y-%m-%d")
            except:
                date_str = "Unknown date"
            caption_preview = (row['caption'][:50] + "...") if len(str(row['caption'])) > 50 else str(row['caption'])
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
        if post_df.empty:
            st.warning("No data for selected post")
            return
            
        post_info = post_df.iloc[0]

        # Media preview
        st.subheader("🎥 Media Preview")

        media_url = post_info.get('media_url')
        thumbnail_url = post_info.get('thumbnail_url')
        media_type = post_info.get('media_type')
        permalink = post_info.get('permalink')

        try:
            if media_type == 'VIDEO' or post_info.get('media_product_type') == 'REELS':
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
            # Try different interaction metrics
            interactions = metrics_map.get('total_interactions', 0)
            if interactions == 0:
                # Calculate from components
                likes = metrics_map.get('likes', 0)
                comments = metrics_map.get('comments', 0)
                shares = metrics_map.get('shares', 0)
                saves = metrics_map.get('saved', 0)
                interactions = likes + comments + shares + saves
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
        try:
            kpis = compute_instagram_kpis(post_df, follower_count)

            if kpis:
                kpi_cols = st.columns(3)

                with kpi_cols[0]:
                    if 'engagement_rate_by_reach' in kpis:
                        st.metric("Engagement Rate (Reach)", f"{kpis['engagement_rate_by_reach']:.2f}%")

                    if 'save_rate' in kpis:
                        st.metric("Save Rate", f"{kpis['save_rate']:.2f}%")

                with kpi_cols[1]:
                    if follower_count:
                        st.metric("Follower Count", f"{follower_count:,}")

                with kpi_cols[2]:
                    # Show available KPIs
                    for key, value in kpis.items():
                        if isinstance(value, (int, float)) and key not in ['follower_count', 'media_count']:
                            if key.endswith('_rate'):
                                st.metric(key.replace('_', ' ').title(), f"{value:.2f}%")
                            else:
                                st.metric(key.replace('_', ' ').title(), f"{value:,}")
        except Exception as e:
            st.warning(f"Could not compute KPIs: {e}")

        # Historical comparison
        st.subheader("📈 Historical Comparison")

        try:
            posting_insights = compute_posting_insights(df)
            if posting_insights:
                comp_cols = st.columns(2)

                with comp_cols[0]:
                    avg_engagement = posting_insights.get('avg_engagement', 0)

                    if avg_engagement > 0:
                        performance_vs_avg = ((interactions - avg_engagement) / avg_engagement) * 100

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
        except Exception as e:
            st.warning(f"Could not compute posting insights: {e}")

    # Caption Analysis
    with st.expander("✍️ Caption Analysis"):
        try:
            caption = str(post_info.get('caption', ''))
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
        except Exception as e:
            st.warning(f"Could not analyze caption: {e}")

    # AI Commentary
    with st.expander("🤖 AI Commentary & Recommendations"):
        if openai_api_key and OPENAI_AVAILABLE:
            try:
                # Prepare context for AI
                if follower_count and follower_count > 0:
                    engagement_rate = ((interactions / follower_count) * 100)
                    engagement_rate_str = f"{engagement_rate:.2f}%"
                else:
                    engagement_rate_str = "N/A"

                context = f"""
                Post Date: {pd.to_datetime(post_info['timestamp']).strftime('%Y-%m-%d') if pd.notna(post_info['timestamp']) else 'Unknown'}
                Media Type: {post_info.get('media_type', 'Unknown')} / {post_info.get('media_product_type', 'Unknown')}
                Caption: {str(post_info.get('caption', ''))[:200] if post_info.get('caption') else 'No caption'}

                Key Metrics:
                - Reach: {metrics_map.get('reach', 0):,}
                - Interactions: {interactions:,}
                - Comments: {metrics_map.get('comments', 0):,}
                - Shares: {metrics_map.get('shares', 0):,}
                - Saves: {metrics_map.get('saved', 0):,}
                - Follower Count: {follower_count:,}

                Engagement Rate: {engagement_rate_str}
                """

                ai_commentary = cached_openai_commentary(selected_media_id, context)
                st.write(ai_commentary)
            except Exception as e:
                st.error(f"Failed to generate AI commentary: {str(e)}")
                logger.error(f"AI commentary error: {e}")
        else:
            if not OPENAI_AVAILABLE:
                st.warning("⚠️ OpenAI package not installed. Install with: `pip install openai`")
            else:
                st.warning("⚠️ OpenAI API key not configured. Set OPENAI_API_KEY environment variable to enable AI commentary.")

    # Trends Over Time
    with st.expander("📈 Instagram Trends Over Time"):
        if len(df['media_id'].unique()) > 1:
            # Metric selector
            selected_metrics = st.multiselect(
                "Select metrics to plot:",
                options=available_metrics,
                default=[m for m in ['reach', 'likes'] if m in available_metrics][:2]
            )

            if selected_metrics:
                try:
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
                except Exception as e:
                    st.warning(f"Could not create trend chart: {e}")
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
            # Add force refresh option
            force_refresh = st.sidebar.checkbox("Force refresh (bypass cache)", value=False)
            
            # Use the optimized function with force refresh option
            paid_data = get_campaign_performance_with_creatives(
                date_preset=date_preset, 
                include_creatives=include_creatives,
                force_refresh=force_refresh
            )

            if paid_data.empty:
                st.warning("No paid campaign data available. This could be due to:")
                st.write("• Rate limiting (please wait and try again)")
                st.write("• Meta Ads account connection issues")
                st.write("• Missing AD_ACCOUNT_ID environment variable")
                st.write("• No campaigns in the selected date range")
                
                # Show API usage stats
                api_stats = get_api_stats()
                if api_stats['total_calls'] > 100:
                    st.warning(f"⚠️ API usage: {api_stats['total_calls']} calls in {api_stats['session_duration_minutes']:.1f} minutes")
                return

            # Display summary metrics
            if len(paid_data) > 0:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    total_spend = pd.to_numeric(paid_data['spend'], errors='coerce').sum()
                    st.metric("Total Spend", f"${total_spend:,.2f}")

                with col2:
                    total_impressions = pd.to_numeric(paid_data['impressions'], errors='coerce').sum()
                    st.metric("Total Impressions", f"{total_impressions:,}")

                with col3:
                    total_clicks = pd.to_numeric(paid_data['clicks'], errors='coerce').sum()
                    st.metric("Total Clicks", f"{total_clicks:,}")

                with col4:
                    avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
                    st.metric("Average CTR", f"{avg_ctr:.2f}%")

            # Show campaign data table
            st.subheader("📊 Campaign Performance Data")

            # Format numeric columns for better display
            display_df = paid_data.copy()
            if 'spend' in display_df.columns:
                display_df['spend'] = pd.to_numeric(display_df['spend'], errors='coerce').round(2)
            if 'ctr' in display_df.columns:
                display_df['ctr'] = pd.to_numeric(display_df['ctr'], errors='coerce').round(3)
            if 'cpc' in display_df.columns:
                display_df['cpc'] = pd.to_numeric(display_df['cpc'], errors='coerce').round(3)

            st.dataframe(display_df, use_container_width=True)

            # Show creative previews if available
            if include_creatives and 'creative_image_url' in paid_data.columns:
                st.subheader("🎨 Creative Previews")

                # Filter rows with creative URLs
                creative_rows = paid_data[paid_data['creative_image_url'].notna()].head(5)

                if not creative_rows.empty:
                    # Show creatives in a grid layout
                    for i, (_, row) in enumerate(creative_rows.iterrows()):
                        with st.expander(f"🎨 {row.get('creative_name', f'Creative {i+1}')} - {row.get('campaign_name', 'Campaign')}"):
                            col1, col2, col3 = st.columns([1, 2, 1])

                            with col1:
                                # Try to display creative image
                                image_displayed = False
                                for url_field in ['creative_image_url', 'creative_thumbnail_url']:
                                    if pd.notna(row.get(url_field)) and not image_displayed:
                                        try:
                                            st.image(row[url_field], caption="Creative Preview", use_column_width=True)
                                            image_displayed = True
                                        except Exception as e:
                                            logger.debug(f"Could not load {url_field}: {e}")
                                            continue
                                
                                if not image_displayed:
                                    st.info("📷 No image preview available")

                            with col2:
                                st.write(f"**Campaign:** {row.get('campaign_name', 'N/A')}")
                                st.write(f"**Ad Name:** {row.get('ad_name', 'N/A')}")
                                
                                if pd.notna(row.get('creative_title')):
                                    st.write(f"**Title:** {row.get('creative_title')}")
                                    
                                if pd.notna(row.get('creative_body')):
                                    body_text = str(row.get('creative_body'))
                                    if len(body_text) > 150:
                                        st.write(f"**Body:** {body_text[:150]}...")
                                        with st.expander("Show full text"):
                                            st.text(body_text)
                                    else:
                                        st.write(f"**Body:** {body_text}")
                                
                                if pd.notna(row.get('creative_object_url')):
                                    st.markdown(f"[🔗 Creative Link]({row.get('creative_object_url')})")

                            with col3:
                                # Show performance metrics for this creative
                                try:
                                    impressions = int(float(row.get('impressions', 0)))
                                    clicks = int(float(row.get('clicks', 0)))
                                    spend = float(row.get('spend', 0))
                                    ctr = float(row.get('ctr', 0))
                                    
                                    st.metric("Impressions", f"{impressions:,}")
                                    st.metric("Clicks", f"{clicks:,}")
                                    st.metric("Spend", f"${spend:.2f}")
                                    st.metric("CTR", f"{ctr:.2f}%")
                                except (ValueError, TypeError) as e:
                                    st.warning("⚠️ Metrics data format error")
                                    logger.debug(f"Metrics error: {e}")
                else:
                    st.info("💡 No creative previews available for current campaigns.")

        except Exception as e:
            st.error(f"Error loading paid campaign data: {str(e)}")
            logger.error(f"Paid campaign error: {e}", exc_info=True)

def validate_environment():
    """Validate environment variables and show status"""
    env_status = {
        'PAGE_ID': bool(os.getenv('PAGE_ID')),
        'IG_USER_ID': bool(os.getenv('IG_USER_ID')),
        'PAGE_ACCESS_TOKEN': bool(os.getenv('PAGE_ACCESS_TOKEN')),
        'META_ACCESS_TOKEN': bool(os.getenv('META_ACCESS_TOKEN')),
        'AD_ACCOUNT_ID': bool(os.getenv('AD_ACCOUNT_ID')),
        'OPENAI_API_KEY': bool(os.getenv('OPENAI_API_KEY')),
        'META_APP_ID': bool(os.getenv('META_APP_ID')),
        'META_APP_SECRET': bool(os.getenv('META_APP_SECRET')),
    }
    
    missing_vars = [var for var, present in env_status.items() if not present]
    
    if missing_vars:
        st.sidebar.warning(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
        with st.sidebar.expander("🔧 Environment Setup Help"):
            st.write("**Required for Instagram insights:**")
            st.code("PAGE_ID, IG_USER_ID, PAGE_ACCESS_TOKEN")
            st.write("**Required for paid campaigns:**")
            st.code("AD_ACCOUNT_ID, META_ACCESS_TOKEN")
            st.write("**Recommended for security:**")
            st.code("META_APP_ID, META_APP_SECRET")
            st.write("**Optional for AI features:**")
            st.code("OPENAI_API_KEY")
            st.write("**How to set in Replit:**")
            st.write("1. Click the 🔒 Secrets tab in the sidebar")
            st.write("2. Add each environment variable")
            st.write("3. Restart your app")
    else:
        st.sidebar.success("✅ All environment variables configured")
    
    return env_status

def main():
    """Main dashboard function"""
    st.title("🚀 AI-Powered Social Campaign Optimizer")
    st.markdown("*Comprehensive Instagram insights with AI-driven recommendations for 2025*")
    
    # Validate environment
    env_status = validate_environment()

    # Create tabs for different sections
    tab1, tab2 = st.tabs(["📸 Instagram Insights", "💰 Paid Campaigns"])

    with tab1:
        show_instagram_insights()

    with tab2:
        show_paid_campaign_insights()

    # API Usage Statistics Section
    with st.expander("📊 API Usage Statistics", expanded=False):
        api_stats = get_api_stats()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total API Calls", api_stats['total_calls'])
        
        with col2:
            st.metric("Calls per Minute", f"{api_stats['calls_per_minute']:.1f}")
        
        with col3:
            st.metric("Session Duration", f"{api_stats['session_duration_minutes']:.1f} min")
        
        # Rate limit warnings
        if api_stats['calls_per_minute'] > 3:
            st.warning("⚠️ High API usage detected. Consider using cached data or reducing fetch frequency.")
        elif api_stats['total_calls'] > 200:
            st.info("ℹ️ High total API usage this session. Monitor for rate limits.")
        else:
            st.success("✅ API usage within normal limits")
        
        st.caption(f"Session started: {api_stats['session_start']}")

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit • Powered by Meta Graph API & OpenAI • Optimized with Rate Limiting*")

if __name__ == "__main__":
    main()
