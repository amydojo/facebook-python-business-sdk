
"""
Enhanced Streamlit dashboard with advanced Instagram analytics, comprehensive metrics, 
KPI computation, trend analysis, and AI-powered insights.
"""
import os
import logging
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="AI Campaign Optimizer - Enhanced Instagram Analytics",
    page_icon="📸",
    layout="wide"
)

def check_environment():
    """Check required environment variables and dependencies."""
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

def check_instagram_env():
    """Check Instagram-specific environment variables."""
    required = ["PAGE_ACCESS_TOKEN", "IG_USER_ID"]
    missing = [k for k in required if not os.getenv(k)]
    
    if missing:
        st.error(f"❌ Missing Instagram environment variables: {missing}")
        st.info("Please set a valid Page Access Token with instagram_basic & instagram_manage_insights permissions, and IG_USER_ID in Replit Secrets.")
        st.stop()

@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_get_paid_with_creatives(date_preset):
    """Cached version of paid campaign fetch with creatives."""
    try:
        from fetch_paid import get_campaign_performance_with_creatives
        return get_campaign_performance_with_creatives(date_preset=date_preset)
    except Exception as e:
        logger.error(f"Error in cached_get_paid_with_creatives: {e}", exc_info=True)
        return pd.DataFrame()

@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_fetch_ig_insights(ig_user_id, since_str, until_str):
    """Cached version of enhanced Instagram insights fetch."""
    try:
        from fetch_organic import fetch_ig_media_insights
        return fetch_ig_media_insights(ig_user_id, since=since_str, until=until_str)
    except Exception as e:
        logger.error(f"Error in cached_fetch_ig_insights: {e}", exc_info=True)
        return pd.DataFrame()

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def cached_get_follower_count(ig_user_id):
    """Cached version of follower count fetch."""
    try:
        from fetch_organic import get_ig_follower_count
        return get_ig_follower_count(ig_user_id)
    except Exception as e:
        logger.error(f"Error in cached_get_follower_count: {e}", exc_info=True)
        return None

def compute_engagement_analysis(df: pd.DataFrame, selected_media_id: str, follower_count: Optional[int]) -> Dict:
    """Compute comprehensive engagement analysis for selected media."""
    sel_data = df[df['media_id'] == selected_media_id]
    metrics_map = sel_data.set_index('metric')['value'].to_dict()
    
    analysis = {
        'metrics': metrics_map,
        'kpis': {},
        'benchmarks': {}
    }
    
    # Core metrics
    total_interactions = metrics_map.get('total_interactions', 0)
    impressions = metrics_map.get('impressions', 0)
    reach = metrics_map.get('reach', 0)
    likes = metrics_map.get('likes', 0)
    comments = metrics_map.get('comments', 0)
    shares = metrics_map.get('shares', 0)
    saves = metrics_map.get('saves', metrics_map.get('saved', 0))
    
    # Engagement rates
    if follower_count and follower_count > 0:
        analysis['kpis']['engagement_rate_followers'] = (total_interactions / follower_count) * 100
    
    if reach and reach > 0:
        analysis['kpis']['engagement_rate_reach'] = (total_interactions / reach) * 100
    
    if impressions and impressions > 0:
        analysis['kpis']['engagement_rate_impressions'] = (total_interactions / impressions) * 100
        analysis['kpis']['save_rate'] = (saves / impressions) * 100
        analysis['kpis']['comment_rate'] = (comments / impressions) * 100
        analysis['kpis']['share_rate'] = (shares / impressions) * 100
    
    # Video/Reels metrics
    video_views = metrics_map.get('video_views', 0)
    plays = metrics_map.get('plays', 0)
    reels_plays = metrics_map.get('ig_reels_plays', 0)
    avg_watch_time = metrics_map.get('ig_reels_avg_watch_time', 0)
    total_watch_time = metrics_map.get('ig_reels_video_view_total_time', 0)
    
    if reach and video_views:
        analysis['kpis']['video_view_rate'] = (video_views / reach) * 100
    
    if reels_plays and avg_watch_time:
        analysis['kpis']['reels_avg_watch_time'] = avg_watch_time
        analysis['kpis']['reels_total_watch_time'] = total_watch_time
    
    # Growth metrics
    profile_visits = metrics_map.get('profile_visits', 0)
    follows = metrics_map.get('follows', 0)
    
    if impressions and impressions > 0:
        analysis['kpis']['profile_visit_rate'] = (profile_visits / impressions) * 100
        analysis['kpis']['follow_rate'] = (follows / impressions) * 100
    
    # Industry benchmarks (approximate)
    analysis['benchmarks'] = {
        'good_engagement_rate': 3.0,  # 3%+ is considered good
        'excellent_engagement_rate': 6.0,  # 6%+ is excellent
        'good_save_rate': 1.0,  # 1%+ save rate is good
        'good_comment_rate': 0.5,  # 0.5%+ comment rate is good
        'good_share_rate': 0.3,  # 0.3%+ share rate is good
    }
    
    return analysis

def analyze_caption_performance(caption: str) -> Dict:
    """Analyze caption characteristics and performance indicators."""
    if not caption:
        return {"word_count": 0, "hashtag_count": 0, "emoji_count": 0, "mentions": 0}
    
    words = caption.split()
    hashtags = [word for word in words if word.startswith('#')]
    mentions = [word for word in words if word.startswith('@')]
    
    # Simple emoji detection (expand as needed)
    emoji_chars = "😀😂😍👍🔥✨💕🎉😊😎❤️💯🙌👏🎵📸🌟💪🏼"
    emoji_count = sum(1 for char in caption if char in emoji_chars)
    
    return {
        "word_count": len(words),
        "hashtag_count": len(hashtags),
        "emoji_count": emoji_count,
        "mentions": len(mentions),
        "character_count": len(caption),
        "hashtags": hashtags[:5],  # Show first 5 hashtags
        "has_cta": any(cta in caption.lower() for cta in ["link in bio", "swipe", "comment", "follow", "tag", "share"])
    }

def generate_ai_insights(analysis: Dict, caption_analysis: Dict, media_type: str) -> str:
    """Generate AI-powered insights and recommendations."""
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_key:
        return "AI insights unavailable - OpenAI API key not configured."
    
    try:
        # Build context for AI analysis
        kpis = analysis.get('kpis', {})
        metrics = analysis.get('metrics', {})
        
        context = f"""
        Instagram {media_type} Performance Analysis:
        
        Engagement Metrics:
        - Total Interactions: {metrics.get('total_interactions', 0):,}
        - Impressions: {metrics.get('impressions', 0):,}
        - Reach: {metrics.get('reach', 0):,}
        - Engagement Rate: {kpis.get('engagement_rate_reach', 0):.2f}%
        - Save Rate: {kpis.get('save_rate', 0):.2f}%
        - Comment Rate: {kpis.get('comment_rate', 0):.2f}%
        
        Caption Analysis:
        - Word Count: {caption_analysis.get('word_count', 0)}
        - Hashtags: {caption_analysis.get('hashtag_count', 0)}
        - Emojis: {caption_analysis.get('emoji_count', 0)}
        - Has CTA: {caption_analysis.get('has_cta', False)}
        
        Video Metrics (if applicable):
        - Video Views: {metrics.get('video_views', 0):,}
        - Plays: {metrics.get('plays', 0):,}
        - Average Watch Time: {kpis.get('reels_avg_watch_time', 0)} seconds
        """
        
        # Use OpenAI to generate insights
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert Instagram marketing analyst. Provide actionable insights and recommendations based on performance data."},
                {"role": "user", "content": f"Analyze this Instagram post performance and provide 3-4 specific, actionable recommendations for improvement:\n\n{context}"}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"AI insights generation failed: {e}")
        return f"AI insights temporarily unavailable: {str(e)}"

def show_advanced_instagram_analytics():
    """Display comprehensive Instagram analytics with advanced features."""
    st.header("📸 Advanced Instagram Analytics & Performance Intelligence")
    
    ig_user_id = os.getenv("IG_USER_ID")
    check_instagram_env()
    
    # Controls in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        since_date = st.date_input(
            "📅 Since", 
            value=(date.today() - timedelta(days=30)),
            max_value=date.today() - timedelta(days=1)
        )
    
    with col2:
        until_date = st.date_input(
            "📅 Until", 
            value=(date.today() - timedelta(days=1)),
            max_value=date.today() - timedelta(days=1)
        )
    
    with col3:
        st.write("")  # Spacing
        fetch_data = st.button("🔄 Fetch Instagram Data", type="primary")
    
    if since_date > until_date:
        st.error("❌ Start date must be before end date")
        return
    
    if fetch_data:
        with st.spinner("🔍 Fetching comprehensive Instagram insights with metadata discovery..."):
            try:
                since_str = since_date.strftime("%Y-%m-%d")
                until_str = until_date.strftime("%Y-%m-%d")
                
                # Fetch data
                df = cached_fetch_ig_insights(ig_user_id, since_str, until_str)
                follower_count = cached_get_follower_count(ig_user_id)
                
                if df.empty:
                    st.warning(f"⚠️ No Instagram data found for {since_str} to {until_str}")
                    st.info("📋 Possible reasons: insufficient permissions, no posts in date range, or API rate limits")
                    return
                
                # Store in session state for use across sections
                st.session_state.ig_data = df
                st.session_state.follower_count = follower_count
                st.session_state.date_range = (since_str, until_str)
                
                st.success(f"✅ Fetched {len(df)} insights records from {len(df['media_id'].unique())} posts")
                
    # Check if we have data to display
    if 'ig_data' not in st.session_state:
        st.info("👆 Click 'Fetch Instagram Data' to load analytics")
        return
    
    df = st.session_state.ig_data
    follower_count = st.session_state.follower_count
    since_str, until_str = st.session_state.date_range
    
    # Quick metrics overview
    metrics_available = sorted(df['metric'].unique())
    unique_posts = len(df['media_id'].unique())
    
    st.markdown("---")
    
    # Overview metrics
    with st.container():
        st.subheader("📊 Analytics Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📱 Posts Analyzed", f"{unique_posts:,}")
        
        with col2:
            st.metric("📈 Metrics Available", f"{len(metrics_available)}")
        
        with col3:
            if follower_count:
                st.metric("👥 Followers", f"{follower_count:,}")
            else:
                st.metric("👥 Followers", "N/A")
        
        with col4:
            st.metric("📅 Date Range", f"{(pd.to_datetime(until_str) - pd.to_datetime(since_str)).days + 1} days")
        
        # Available metrics info
        with st.expander("🔍 Available Metrics Discovered", expanded=False):
            st.info(f"**Discovered Metrics:** {', '.join(metrics_available)}")
            
            # Categorize metrics
            engagement_metrics = [m for m in metrics_available if any(keyword in m for keyword in ['likes', 'comments', 'shares', 'saves', 'interactions'])]
            video_metrics = [m for m in metrics_available if any(keyword in m for keyword in ['video', 'plays', 'reels', 'watch'])]
            reach_metrics = [m for m in metrics_available if any(keyword in m for keyword in ['impressions', 'reach'])]
            growth_metrics = [m for m in metrics_available if any(keyword in m for keyword in ['profile', 'follows', 'website'])]
            
            col1, col2 = st.columns(2)
            with col1:
                if engagement_metrics:
                    st.write("**🎯 Engagement:** " + ", ".join(engagement_metrics))
                if video_metrics:
                    st.write("**🎬 Video/Reels:** " + ", ".join(video_metrics))
            with col2:
                if reach_metrics:
                    st.write("**📢 Reach:** " + ", ".join(reach_metrics))
                if growth_metrics:
                    st.write("**📈 Growth:** " + ", ".join(growth_metrics))
    
    st.markdown("---")
    
    # Post selector
    with st.container():
        st.subheader("🎯 Individual Post Analysis")
        
        # Build post options
        unique_media = df[['media_id', 'timestamp', 'caption', 'media_url', 'permalink', 'thumbnail_url', 'media_type', 'media_product_type']].drop_duplicates(subset=['media_id'])
        
        if unique_media.empty:
            st.warning("No unique media found")
            return
        
        post_options = {}
        for _, row in unique_media.iterrows():
            media_id = row['media_id']
            timestamp = row['timestamp']
            caption = row.get('caption', '')
            media_type = row.get('media_type', 'UNKNOWN')
            product_type = row.get('media_product_type', '')
            
            # Create readable label
            date_part = timestamp.split('T')[0] if timestamp else 'Unknown date'
            caption_snippet = caption[:50] + "..." if caption and len(caption) > 50 else caption or 'No caption'
            type_label = f"{media_type}" + (f"/{product_type}" if product_type else "")
            
            post_options[media_id] = f"📅 {date_part} | {type_label} | {caption_snippet}"
        
        selected_media_id = st.selectbox(
            "Select post to analyze:",
            options=list(post_options.keys()),
            format_func=lambda x: post_options[x],
            key="post_selector"
        )
        
        # Get selected media info
        selected_media = unique_media[unique_media['media_id'] == selected_media_id].iloc[0]
        
        # Display selected post
        col_preview, col_analysis = st.columns([1, 2])
        
        with col_preview:
            st.markdown("### 🖼️ Preview")
            
            # Media preview
            media_url = selected_media.get('media_url') or selected_media.get('thumbnail_url')
            if media_url and pd.notna(media_url):
                try:
                    # Determine media type for appropriate display
                    if any(ext in media_url.lower() for ext in ['.mp4', '.mov', '.webm']) or selected_media.get('media_type') == 'VIDEO':
                        st.video(media_url)
                    else:
                        st.image(media_url, use_column_width=True)
                except Exception as e:
                    st.warning(f"Could not load media: {e}")
                    st.info("📱 Media preview unavailable")
            else:
                st.info("📱 No preview available")
            
            # Instagram link
            permalink = selected_media.get('permalink')
            if permalink and pd.notna(permalink):
                st.markdown(f"🔗 [View on Instagram]({permalink})")
            
            # Basic info
            st.markdown(f"**📅 Posted:** {selected_media.get('timestamp', '').split('T')[0]}")
            st.markdown(f"**📱 Type:** {selected_media.get('media_type', 'Unknown')}")
            if selected_media.get('media_product_type'):
                st.markdown(f"**🏷️ Product:** {selected_media.get('media_product_type')}")
        
        with col_analysis:
            st.markdown("### 📊 Performance Metrics")
            
            # Get metrics for this post
            post_metrics = df[df['media_id'] == selected_media_id][['metric', 'value']].set_index('metric')['value'].to_dict()
            
            if post_metrics:
                # Display key metrics in columns
                metric_cols = st.columns(3)
                
                key_metrics = [
                    ('impressions', 'Impressions', '👁️'),
                    ('reach', 'Reach', '📢'),
                    ('total_interactions', 'Interactions', '❤️'),
                    ('likes', 'Likes', '👍'),
                    ('comments', 'Comments', '💬'),
                    ('shares', 'Shares', '↗️'),
                    ('saves', 'Saves', '🔖'),
                    ('video_views', 'Video Views', '▶️'),
                    ('profile_visits', 'Profile Visits', '👤'),
                    ('follows', 'Follows', '➕')
                ]
                
                displayed_count = 0
                for metric_key, label, icon in key_metrics:
                    if metric_key in post_metrics:
                        value = post_metrics[metric_key]
                        col_idx = displayed_count % 3
                        metric_cols[col_idx].metric(f"{icon} {label}", f"{int(value):,}" if value else "0")
                        displayed_count += 1
                        
                        if displayed_count >= 9:  # Limit to 9 metrics (3 rows)
                            break
                
                # Show all metrics in expandable section
                with st.expander("📋 All Metrics", expanded=False):
                    metrics_df = pd.DataFrame([
                        {"Metric": metric, "Value": f"{int(value):,}" if value else "0"}
                        for metric, value in post_metrics.items()
                    ])
                    st.dataframe(metrics_df, use_container_width=True)
            else:
                st.warning("No metrics available for this post")
    
    st.markdown("---")
    
    # Advanced KPI Analysis
    with st.container():
        st.subheader("🎯 Advanced KPI Analysis")
        
        # Compute comprehensive analysis
        analysis = compute_engagement_analysis(df, selected_media_id, follower_count)
        kpis = analysis['kpis']
        benchmarks = analysis['benchmarks']
        
        if kpis:
            # Engagement rates
            st.markdown("#### 📈 Engagement Analysis")
            
            rate_cols = st.columns(4)
            
            # Engagement rate (primary metric)
            eng_rate = kpis.get('engagement_rate_reach') or kpis.get('engagement_rate_impressions') or kpis.get('engagement_rate_followers')
            if eng_rate is not None:
                # Color code based on benchmark
                if eng_rate >= benchmarks['excellent_engagement_rate']:
                    delta_color = "normal"
                    status = "🔥 Excellent"
                elif eng_rate >= benchmarks['good_engagement_rate']:
                    delta_color = "normal" 
                    status = "✅ Good"
                else:
                    delta_color = "inverse"
                    status = "⚠️ Below Average"
                
                rate_cols[0].metric(
                    "📊 Engagement Rate", 
                    f"{eng_rate:.2f}%",
                    delta=status
                )
            
            # Other rates
            if 'save_rate' in kpis:
                save_rate = kpis['save_rate']
                save_status = "✅ Good" if save_rate >= benchmarks['good_save_rate'] else "⚠️ Low"
                rate_cols[1].metric("🔖 Save Rate", f"{save_rate:.2f}%", delta=save_status)
            
            if 'comment_rate' in kpis:
                comment_rate = kpis['comment_rate']
                comment_status = "✅ Good" if comment_rate >= benchmarks['good_comment_rate'] else "⚠️ Low"
                rate_cols[2].metric("💬 Comment Rate", f"{comment_rate:.2f}%", delta=comment_status)
            
            if 'share_rate' in kpis:
                share_rate = kpis['share_rate']
                share_status = "✅ Good" if share_rate >= benchmarks['good_share_rate'] else "⚠️ Low"
                rate_cols[3].metric("↗️ Share Rate", f"{share_rate:.2f}%", delta=share_status)
            
            # Video/Reels specific KPIs
            if 'reels_avg_watch_time' in kpis or 'video_view_rate' in kpis:
                st.markdown("#### 🎬 Video Performance")
                
                video_cols = st.columns(3)
                
                if 'video_view_rate' in kpis:
                    video_cols[0].metric("▶️ Video View Rate", f"{kpis['video_view_rate']:.2f}%")
                
                if 'reels_avg_watch_time' in kpis:
                    video_cols[1].metric("⏱️ Avg Watch Time", f"{kpis['reels_avg_watch_time']:.1f}s")
                
                if 'reels_total_watch_time' in kpis:
                    total_watch = kpis['reels_total_watch_time']
                    hours = total_watch // 3600
                    minutes = (total_watch % 3600) // 60
                    video_cols[2].metric("🕒 Total Watch Time", f"{int(hours)}h {int(minutes)}m")
            
            # Growth metrics
            if 'profile_visit_rate' in kpis or 'follow_rate' in kpis:
                st.markdown("#### 📈 Growth Impact")
                
                growth_cols = st.columns(2)
                
                if 'profile_visit_rate' in kpis:
                    growth_cols[0].metric("👤 Profile Visit Rate", f"{kpis['profile_visit_rate']:.3f}%")
                
                if 'follow_rate' in kpis:
                    growth_cols[1].metric("➕ Follow Rate", f"{kpis['follow_rate']:.3f}%")
        
        else:
            st.info("📊 KPI analysis requires impression/reach data")
    
    st.markdown("---")
    
    # Caption Analysis
    with st.container():
        st.subheader("📝 Caption Performance Analysis")
        
        caption = selected_media.get('caption', '')
        caption_analysis = analyze_caption_performance(caption)
        
        if caption:
            # Display caption
            with st.expander("📖 Full Caption", expanded=False):
                st.write(caption)
            
            # Caption metrics
            caption_cols = st.columns(5)
            
            caption_cols[0].metric("📝 Words", caption_analysis['word_count'])
            caption_cols[1].metric("#️⃣ Hashtags", caption_analysis['hashtag_count'])
            caption_cols[2].metric("😊 Emojis", caption_analysis['emoji_count'])
            caption_cols[3].metric("@ Mentions", caption_analysis['mentions'])
            caption_cols[4].metric("🎯 Has CTA", "Yes" if caption_analysis['has_cta'] else "No")
            
            # Hashtag analysis
            if caption_analysis['hashtags']:
                st.markdown("**🏷️ Top Hashtags:** " + " ".join(caption_analysis['hashtags']))
        else:
            st.info("No caption available for this post")
    
    st.markdown("---")
    
    # AI-Powered Insights
    with st.container():
        st.subheader("🧠 AI-Powered Insights & Recommendations")
        
        if st.button("🔮 Generate AI Analysis", type="secondary"):
            with st.spinner("🤖 Analyzing performance with AI..."):
                ai_insights = generate_ai_insights(
                    analysis, 
                    caption_analysis, 
                    selected_media.get('media_type', 'Unknown')
                )
                
                st.markdown("#### 💡 AI Recommendations")
                st.write(ai_insights)
    
    st.markdown("---")
    
    # Trend Analysis
    with st.container():
        st.subheader("📈 Performance Trends & Comparisons")
        
        # Historical comparison
        if unique_posts > 1:
            st.markdown("#### 📊 Historical Performance")
            
            # Compute engagement rates for all posts
            historical_data = []
            for media_id in df['media_id'].unique():
                media_metrics = df[df['media_id'] == media_id].set_index('metric')['value'].to_dict()
                media_info = unique_media[unique_media['media_id'] == media_id].iloc[0]
                
                total_interactions = media_metrics.get('total_interactions', 0)
                reach = media_metrics.get('reach', 0)
                impressions = media_metrics.get('impressions', 0)
                
                # Calculate engagement rate
                if reach > 0:
                    eng_rate = (total_interactions / reach) * 100
                elif impressions > 0:
                    eng_rate = (total_interactions / impressions) * 100
                elif follower_count and follower_count > 0:
                    eng_rate = (total_interactions / follower_count) * 100
                else:
                    eng_rate = 0
                
                historical_data.append({
                    'media_id': media_id,
                    'date': media_info['timestamp'].split('T')[0],
                    'media_type': media_info.get('media_type', 'Unknown'),
                    'engagement_rate': eng_rate,
                    'total_interactions': total_interactions,
                    'reach': reach,
                    'impressions': impressions
                })
            
            hist_df = pd.DataFrame(historical_data)
            
            if not hist_df.empty and len(hist_df) > 1:
                # Performance comparison
                current_eng_rate = hist_df[hist_df['media_id'] == selected_media_id]['engagement_rate'].iloc[0]
                avg_eng_rate = hist_df['engagement_rate'].mean()
                median_eng_rate = hist_df['engagement_rate'].median()
                
                comparison_cols = st.columns(3)
                
                comparison_cols[0].metric(
                    "🎯 This Post", 
                    f"{current_eng_rate:.2f}%",
                    delta=f"{current_eng_rate - avg_eng_rate:.2f}% vs avg"
                )
                comparison_cols[1].metric("📊 Average", f"{avg_eng_rate:.2f}%")
                comparison_cols[2].metric("📈 Median", f"{median_eng_rate:.2f}%")
                
                # Trend chart
                fig = go.Figure()
                
                # Add trend line
                fig.add_trace(go.Scatter(
                    x=hist_df['date'],
                    y=hist_df['engagement_rate'],
                    mode='lines+markers',
                    name='Engagement Rate',
                    line=dict(width=3, color='#1f77b4'),
                    marker=dict(size=8)
                ))
                
                # Highlight selected post
                selected_point = hist_df[hist_df['media_id'] == selected_media_id]
                fig.add_trace(go.Scatter(
                    x=selected_point['date'],
                    y=selected_point['engagement_rate'],
                    mode='markers',
                    name='Selected Post',
                    marker=dict(size=15, color='red', symbol='star')
                ))
                
                fig.update_layout(
                    title="📈 Engagement Rate Trend",
                    xaxis_title="Date",
                    yaxis_title="Engagement Rate (%)",
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Best performing content analysis
                best_post = hist_df.loc[hist_df['engagement_rate'].idxmax()]
                if best_post['media_id'] != selected_media_id:
                    st.info(f"🏆 **Best performing post:** {best_post['date']} with {best_post['engagement_rate']:.2f}% engagement rate")
        
        # Aggregate metrics over time
        with st.expander("📊 Detailed Metrics Trends", expanded=False):
            trend_metrics = st.multiselect(
                "Select metrics to visualize:",
                options=metrics_available,
                default=[m for m in ['impressions', 'reach', 'total_interactions'] if m in metrics_available]
            )
            
            if trend_metrics:
                # Aggregate by date
                df_trends = df.copy()
                df_trends['date'] = pd.to_datetime(df_trends['timestamp']).dt.date
                
                trend_pivot = df_trends.pivot_table(
                    index='date',
                    columns='metric',
                    values='value',
                    aggfunc='sum'
                ).reset_index()
                
                # Create trend chart
                fig_trends = go.Figure()
                
                for metric in trend_metrics:
                    if metric in trend_pivot.columns:
                        fig_trends.add_trace(go.Scatter(
                            x=trend_pivot['date'],
                            y=trend_pivot[metric],
                            mode='lines+markers',
                            name=metric.replace('_', ' ').title(),
                            line=dict(width=2)
                        ))
                
                fig_trends.update_layout(
                    title="📈 Metrics Trends Over Time",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig_trends, use_container_width=True)

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

def main():
    """Main dashboard function."""
    logger.info("🚀 Starting Enhanced Instagram Analytics Dashboard")

    # Environment check
    env_check = check_environment()

    st.title("📸 AI-Powered Instagram Analytics & Campaign Intelligence")
    st.markdown("Advanced Instagram performance analytics with AI-powered insights, comprehensive KPIs, and metadata-driven metrics discovery")

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
        from fetch_organic import fetch_ig_media_insights, get_ig_follower_count, compute_instagram_kpis
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
            "PAGE_ACCESS_TOKEN": bool(os.getenv("PAGE_ACCESS_TOKEN")),
            "IG_USER_ID": bool(os.getenv("IG_USER_ID")),
            "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY"))
        }

        for var, is_set in env_vars.items():
            if is_set:
                st.success(f"✅ {var}")
            else:
                st.warning(f"⚠️ {var}")
        
        # Clear cache button
        if st.button("🔄 Clear Cache"):
            st.cache_data.clear()
            st.success("✅ Cache cleared!")

    # Main content tabs
    tab1, tab2 = st.tabs(["📸 Instagram Analytics", "🎯 Paid Campaigns"])

    with tab1:
        show_advanced_instagram_analytics()

    with tab2:
        show_paid_section()

    # Footer
    st.markdown("---")
    st.markdown("🚀 **Enhanced Instagram Analytics** - Built with ❤️ on Replit")
    st.markdown("**🔧 Features:** Metadata-driven metrics discovery • Advanced KPI computation • AI-powered insights • Trend analysis • Performance benchmarking")
    st.markdown("**📚 Documentation:**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- [Instagram Business API](https://developers.facebook.com/docs/instagram-api/)")
        st.markdown("- [Graph API Insights](https://developers.facebook.com/docs/graph-api/reference/insights/)")
    with col2:
        st.markdown("- [Streamlit Documentation](https://docs.streamlit.io/)")
        st.markdown("- [OpenAI API](https://platform.openai.com/docs/)")

if __name__ == "__main__":
    main()
