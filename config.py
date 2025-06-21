
import os
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('campaign_optimizer.log')
    ]
)

logger = logging.getLogger(__name__)

class Config:
    """Configuration management with environment validation"""
    
    def __init__(self):
        self.validate_environment()
        
    def validate_environment(self) -> Dict[str, Any]:
        """Validate and return environment configuration"""
        required_vars = {
            'META_ACCESS_TOKEN': os.getenv('META_ACCESS_TOKEN'),
            'AD_ACCOUNT_ID': os.getenv('AD_ACCOUNT_ID'),
        }
        
        optional_vars = {
            'PAGE_ID': os.getenv('PAGE_ID'),
            'IG_USER_ID': os.getenv('IG_USER_ID'),
            'PAGE_ACCESS_TOKEN': os.getenv('PAGE_ACCESS_TOKEN'),
            'META_APP_ID': os.getenv('META_APP_ID'),
            'META_APP_SECRET': os.getenv('META_APP_SECRET'),
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        }
        
        # Check required variables
        missing_required = [k for k, v in required_vars.items() if not v]
        if missing_required:
            logger.error(f"❌ Missing required environment variables: {missing_required}")
        
        # Log optional variables status
        missing_optional = [k for k, v in optional_vars.items() if not v]
        if missing_optional:
            logger.warning(f"⚠️ Missing optional environment variables: {missing_optional}")
        
        # Log successful configuration
        available_vars = [k for k, v in {**required_vars, **optional_vars}.items() if v]
        logger.info(f"✅ Available environment variables: {available_vars}")
        
        return {
            'required': required_vars,
            'optional': optional_vars,
            'missing_required': missing_required,
            'missing_optional': missing_optional,
            'all_configured': len(missing_required) == 0
        }
    
    @property
    def meta_access_token(self) -> str:
        return os.getenv('META_ACCESS_TOKEN', '')
    
    @property
    def ad_account_id(self) -> str:
        return os.getenv('AD_ACCOUNT_ID', '')
    
    @property
    def page_access_token(self) -> str:
        return os.getenv('PAGE_ACCESS_TOKEN', os.getenv('META_ACCESS_TOKEN', ''))
    
    @property
    def openai_api_key(self) -> str:
        return os.getenv('OPENAI_API_KEY', '')

# Global config instance
config = Config()
