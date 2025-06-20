"""
Facebook Business SDK client initialization.
Reference: https://developers.facebook.com/docs/business-sdk/getting-started/
"""
import logging
from config import config

# Import Facebook Business SDK components after fixing circular imports
try:
    from facebook_business.api import FacebookAdsApi
    from facebook_business.adobjects.adaccount import AdAccount
    from facebook_business.exceptions import FacebookRequestError
    from facebook_business.adobjects.page import Page
except ImportError as e:
    logging.error(f"Facebook Business SDK import error: {e}")
    # Fallback or alternative handling
    FacebookAdsApi = None
    AdAccount = None
    FacebookRequestError = None
    Page = None

logger = logging.getLogger(__name__)

class FacebookClient:
    """Facebook Business SDK client wrapper."""

    def __init__(self):
        self.api = None
        self.ad_account = None
        self._initialize()

    def _initialize(self):
        """
        Initialize Facebook Ads API with app secret proof.
        Reference: https://developers.facebook.com/docs/business-sdk/getting-started/
        """
        try:
            # Validate configuration
            config.validate_required_configs()

            # Check if Facebook SDK was imported successfully
            if FacebookAdsApi is None:
                logger.error("Facebook Business SDK not available due to import errors")
                raise ImportError("Facebook Business SDK not properly imported")

            # Initialize the API with app secret proof for security
            self.api = FacebookAdsApi.init(
                app_id=config.META_APP_ID,
                app_secret=config.META_APP_SECRET,
                access_token=config.META_ACCESS_TOKEN,
                api_version=config.GRAPH_API_VERSION
            )

            # Get ad account
            ad_account_id = config.get_ad_account_id_formatted()
            if ad_account_id:
                self.ad_account = AdAccount(ad_account_id)
                logger.info(f"✅ Facebook API initialized successfully for account: {ad_account_id}")
            else:
                logger.warning("No AD_ACCOUNT_ID provided - some features will be limited")

        except FacebookRequestError as e:
            logger.error(f"❌ Facebook API request error during initialization: {e}")
            self.api = None
            self.ad_account = None
        except Exception as e:
            logger.error(f"❌ Unexpected error during Facebook API initialization: {e}")
            self.api = None
            self.ad_account = None

    def get_ad_account(self):
        """Return the ad account object or None if not available."""
        return self.ad_account

    def is_initialized(self):
        """Check if the client is properly initialized."""
        return self.api is not None and self.ad_account is not None

    def test_connection(self):
        """
        Test the connection by fetching basic account info.
        Returns: dict with connection status and account info
        """
        if not self.is_initialized():
            return {"success": False, "error": "Client not initialized"}

        try:
            # Test connection by getting account info
            account_info = self.ad_account.api_get(fields=['name', 'account_status', 'currency'])
            return {
                "success": True,
                "account_name": account_info.get('name'),
                "account_status": account_info.get('account_status'),
                "currency": account_info.get('currency'),
                "account_id": self.ad_account.get_id()
            }
        except FacebookRequestError as e:
            logger.error(f"Connection test failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error during connection test: {e}")
            return {"success": False, "error": str(e)}

# Global client instance
fb_client = FacebookClient()