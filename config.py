
"""
Configuration module for AI-powered social campaign optimizer.
Manages environment variables and feature flags.
"""
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Config:
    """Configuration class with environment variables and feature flags."""
    
    # Meta/Facebook credentials
    META_APP_ID = os.getenv('META_APP_ID')
    META_APP_SECRET = os.getenv('META_APP_SECRET')
    META_ACCESS_TOKEN = os.getenv('META_ACCESS_TOKEN')
    AD_ACCOUNT_ID = os.getenv('AD_ACCOUNT_ID')
    PAGE_ID = os.getenv('PAGE_ID')
    IG_USER_ID = os.getenv('IG_USER_ID')
    
    # OpenAI credentials
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Airtable credentials
    AIRTABLE_API_KEY = os.getenv('AIRTABLE_API_KEY')
    AIRTABLE_BASE_ID = os.getenv('AIRTABLE_BASE_ID')
    AIRTABLE_LEADS_TABLE = os.getenv('AIRTABLE_LEADS_TABLE', 'Leads')
    AIRTABLE_TRANSACTIONS_TABLE = os.getenv('AIRTABLE_TRANSACTIONS_TABLE', 'Transactions')
    AIRTABLE_AUDIT_TABLE = os.getenv('AIRTABLE_AUDIT_TABLE', 'Audit_Log')
    
    # Feature flags
    ENABLE_ATTRIBUTION = os.getenv('ENABLE_ATTRIBUTION', 'True').lower() == 'true'
    ENABLE_CHAT_INSIGHTS = os.getenv('ENABLE_CHAT_INSIGHTS', 'True').lower() == 'true'
    ENABLE_AUTO_ACTIONS = os.getenv('ENABLE_AUTO_ACTIONS', 'False').lower() == 'true'
    FORCE_MARKOV = os.getenv('FORCE_MARKOV', 'False').lower() == 'true'
    
    # API configuration
    GRAPH_API_VERSION = os.getenv('GRAPH_API_VERSION', 'v18.0')
    
    @classmethod
    def validate_required_configs(cls):
        """Validate that required configuration values are present."""
        required_configs = {
            'META_APP_ID': cls.META_APP_ID,
            'META_APP_SECRET': cls.META_APP_SECRET,
            'META_ACCESS_TOKEN': cls.META_ACCESS_TOKEN,
            'AD_ACCOUNT_ID': cls.AD_ACCOUNT_ID,
        }
        
        missing = [key for key, value in required_configs.items() if not value]
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")
        
        return True
    
    @classmethod
    def get_ad_account_id_formatted(cls):
        """Return properly formatted ad account ID with 'act_' prefix."""
        if not cls.AD_ACCOUNT_ID:
            return None
        
        if cls.AD_ACCOUNT_ID.startswith('act_'):
            return cls.AD_ACCOUNT_ID
        return f"act_{cls.AD_ACCOUNT_ID}"

# Global config instance
config = Config()
