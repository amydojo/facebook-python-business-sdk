
"""
OpenAI integration for AI-powered insights, copy generation, and image creation.
References:
- OpenAI Chat Completion API: https://platform.openai.com/docs/api-reference/chat/create
- OpenAI Images API: https://platform.openai.com/docs/api-reference/images/create
"""
import logging
import os
from config import config

logger = logging.getLogger(__name__)

# Try to import OpenAI with proper error handling
try:
    import openai
    OPENAI_AVAILABLE = True
    
    # Initialize OpenAI API
    openai_api_key = config.openai_api_key if hasattr(config, 'openai_api_key') else os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        openai.api_key = openai_api_key
    else:
        logger.warning("OPENAI_API_KEY not configured - AI features will be disabled")
        OPENAI_AVAILABLE = False
        
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None
    logger.warning("OpenAI package not available - AI features will be disabled")

def call_chat(messages, model="gpt-4", max_tokens=None, temperature=0.7, **kwargs):
    """
    Call OpenAI Chat Completion API.
    
    Args:
        messages: list of message dicts with 'role' and 'content'
        model: OpenAI model name
        max_tokens: maximum tokens in response
        temperature: creativity level (0-1)
        **kwargs: additional parameters
    
    Returns:
        str: AI response content or None if error
        
    Reference: https://platform.openai.com/docs/api-reference/chat/create
    """
    if not OPENAI_AVAILABLE or not openai:
        logger.error("OpenAI not available")
        return None
    
    openai_api_key = config.openai_api_key if hasattr(config, 'openai_api_key') else os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OpenAI API key not configured")
        return None
    
    try:
        logger.info(f"Calling OpenAI Chat API with model {model}")
        
        # Use the new OpenAI client interface
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        content = response.choices[0].message.content
        logger.info("Successfully received OpenAI response")
        return content
        
    except Exception as e:
        error_msg = str(e).lower()
        if 'authentication' in error_msg:
            logger.error("OpenAI authentication failed - check API key")
        elif 'rate limit' in error_msg:
            logger.error("OpenAI rate limit exceeded")
        else:
            logger.error(f"OpenAI API error: {e}")
        return None

def get_embedding(text, model="text-embedding-ada-002"):
    """
    Get text embedding from OpenAI.
    
    Args:
        text: text to embed
        model: embedding model name
    
    Returns:
        list: embedding vector or None if error
    """
    if not OPENAI_AVAILABLE or not openai:
        logger.error("OpenAI not available")
        return None
    
    openai_api_key = config.openai_api_key if hasattr(config, 'openai_api_key') else os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OpenAI API key not configured")
        return None
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)
        
        response = client.embeddings.create(
            model=model,
            input=text
        )
        
        embedding = response.data[0].embedding
        logger.info(f"Generated embedding for text (length: {len(text)})")
        return embedding
        
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def generate_image(prompt_text, size="1024x1024", n=1):
    """
    Generate image using OpenAI DALL-E.
    
    Args:
        prompt_text: image generation prompt
        size: image size ("256x256", "512x512", "1024x1024")
        n: number of images to generate
    
    Returns:
        list: image URLs or None if error
        
    Reference: https://platform.openai.com/docs/api-reference/images/create
    """
    if not OPENAI_AVAILABLE or not openai:
        logger.error("OpenAI not available")
        return None
    
    openai_api_key = config.openai_api_key if hasattr(config, 'openai_api_key') else os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OpenAI API key not configured")
        return None
    
    try:
        logger.info(f"Generating image with prompt: {prompt_text[:100]}...")
        
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)
        
        response = client.images.generate(
            prompt=prompt_text,
            n=n,
            size=size
        )
        
        image_urls = [img.url for img in response.data]
        logger.info(f"Successfully generated {len(image_urls)} images")
        return image_urls
        
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None

def generate_insights_summary(data_summary, metrics_context=""):
    """
    Generate AI-powered insights summary from data.
    
    Args:
        data_summary: str or dict with key metrics
        metrics_context: additional context about the metrics
    
    Returns:
        str: AI-generated insights
    """
    try:
        # Prepare context
        if isinstance(data_summary, dict):
            data_text = "\n".join([f"- {k}: {v}" for k, v in data_summary.items()])
        else:
            data_text = str(data_summary)
        
        prompt = f"""
        Analyze the following marketing performance data and provide actionable insights:
        
        PERFORMANCE DATA:
        {data_text}
        
        {f"CONTEXT: {metrics_context}" if metrics_context else ""}
        
        Please provide:
        1. Key performance highlights
        2. Areas of concern or underperformance
        3. Specific optimization recommendations
        4. Next steps to improve results
        
        Keep the analysis practical and actionable for a marketing team.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert digital marketing analyst with deep expertise in performance optimization and data-driven insights."},
            {"role": "user", "content": prompt}
        ]
        
        return call_chat(messages, max_tokens=400, temperature=0.7)
        
    except Exception as e:
        logger.error(f"Error generating insights summary: {e}")
        return "Unable to generate insights summary."

def generate_ad_copy(brief, tone="professional", audience="general", num_variants=5):
    """
    Generate Facebook ad copy variants.
    
    Args:
        brief: creative brief or product description
        tone: writing tone (professional, casual, urgent, etc.)
        audience: target audience description
        num_variants: number of copy variants to generate
    
    Returns:
        list: ad copy variants
    """
    try:
        prompt = f"""
        Generate {num_variants} Facebook ad copy variants based on this brief:
        
        BRIEF: {brief}
        TONE: {tone}
        TARGET AUDIENCE: {audience}
        
        Requirements:
        - Each variant should be 125 characters or less for primary text
        - Include a clear call-to-action
        - Make each variant distinctly different in approach
        - Optimize for engagement and conversions
        - Follow Facebook advertising best practices
        
        Format as numbered list with just the ad copy text.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert Facebook ads copywriter with a track record of creating high-converting ad copy."},
            {"role": "user", "content": prompt}
        ]
        
        response = call_chat(messages, max_tokens=300, temperature=0.8)
        
        if response:
            # Parse the response into individual variants
            variants = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering/bullets
                    clean_line = line.split('.', 1)[-1].strip()
                    clean_line = clean_line.lstrip('- ')
                    if clean_line:
                        variants.append(clean_line)
            
            logger.info(f"Generated {len(variants)} ad copy variants")
            return variants
        
        return []
        
    except Exception as e:
        logger.error(f"Error generating ad copy: {e}")
        return []

def explain_anomaly(metric_name, change_percent, current_value, previous_value, context=""):
    """
    Generate AI explanation for metric anomalies.
    
    Args:
        metric_name: name of the metric that changed
        change_percent: percentage change
        current_value: current metric value
        previous_value: previous metric value
        context: additional context about the campaign/timeframe
    
    Returns:
        str: AI explanation and recommendations
    """
    try:
        direction = "increased" if change_percent > 0 else "decreased"
        
        prompt = f"""
        A marketing metric has shown an unusual change that needs explanation:
        
        METRIC: {metric_name}
        CHANGE: {direction} by {abs(change_percent):.1f}%
        CURRENT VALUE: {current_value}
        PREVIOUS VALUE: {previous_value}
        {f"CONTEXT: {context}" if context else ""}
        
        Please provide:
        1. Possible causes for this change
        2. Whether this is likely positive, negative, or neutral
        3. Immediate actions to take
        4. What to monitor going forward
        
        Be specific and actionable in your recommendations.
        """
        
        messages = [
            {"role": "system", "content": "You are a digital marketing expert specializing in performance analysis and troubleshooting."},
            {"role": "user", "content": prompt}
        ]
        
        return call_chat(messages, max_tokens=300, temperature=0.7)
        
    except Exception as e:
        logger.error(f"Error explaining anomaly: {e}")
        return f"Unable to analyze the {change_percent:+.1f}% change in {metric_name}. Please review manually."

def generate_creative_ideas(product_type, campaign_goal, target_audience, brand_voice=""):
    """
    Generate creative ideas for ad campaigns.
    
    Args:
        product_type: type of product/service
        campaign_goal: campaign objective (awareness, conversions, etc.)
        target_audience: audience description
        brand_voice: brand voice/personality
    
    Returns:
        str: creative ideas and suggestions
    """
    try:
        prompt = f"""
        Generate creative campaign ideas for:
        
        PRODUCT/SERVICE: {product_type}
        CAMPAIGN GOAL: {campaign_goal}
        TARGET AUDIENCE: {target_audience}
        {f"BRAND VOICE: {brand_voice}" if brand_voice else ""}
        
        Provide:
        1. 3 unique creative concepts
        2. Visual direction for each concept
        3. Key messaging angles
        4. Platform-specific adaptations (Facebook vs Instagram)
        5. Content formats to test (video, carousel, single image)
        
        Focus on innovative, engaging approaches that stand out in the feed.
        """
        
        messages = [
            {"role": "system", "content": "You are a creative director specializing in social media advertising with expertise in Facebook and Instagram campaigns."},
            {"role": "user", "content": prompt}
        ]
        
        return call_chat(messages, max_tokens=500, temperature=0.8)
        
    except Exception as e:
        logger.error(f"Error generating creative ideas: {e}")
        return "Unable to generate creative ideas at this time."

def optimize_audience_targeting(current_audience, performance_data, goal="conversions"):
    """
    Generate audience optimization suggestions.
    
    Args:
        current_audience: description of current targeting
        performance_data: performance metrics context
        goal: optimization goal
    
    Returns:
        str: audience optimization recommendations
    """
    try:
        prompt = f"""
        Analyze and optimize audience targeting:
        
        CURRENT TARGETING: {current_audience}
        PERFORMANCE DATA: {performance_data}
        OPTIMIZATION GOAL: {goal}
        
        Provide:
        1. Analysis of current targeting effectiveness
        2. Specific audience expansion opportunities
        3. Audience segments to test
        4. Lookalike audience recommendations
        5. Interest and behavior targeting suggestions
        
        Focus on data-driven recommendations that align with the optimization goal.
        """
        
        messages = [
            {"role": "system", "content": "You are a Facebook ads targeting specialist with expertise in audience optimization and segmentation."},
            {"role": "user", "content": prompt}
        ]
        
        return call_chat(messages, max_tokens=400, temperature=0.7)
        
    except Exception as e:
        logger.error(f"Error optimizing audience targeting: {e}")
        return "Unable to generate targeting recommendations at this time."
