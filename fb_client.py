"""
Facebook Business SDK client initialization.
Reference: https://developers.facebook.com/docs/business-sdk/getting-started/
"""
import logging
from config import config

# Import Facebook Business SDK components with proper error handling
FacebookAdsApi = None
AdAccount = None
FacebookRequestError = None
Page = None

logger = logging.getLogger('fb_client')

# Check if facebook_business module is available at all
try:
    import facebook_business
    logger.info(f"✅ Facebook Business SDK base module available (version: {getattr(facebook_business, '__version__', 'unknown')})")
    
    # Check for circular import by inspecting the api module
    try:
        import facebook_business.api
        logger.info("✅ facebook_business.api module accessible")
    except Exception as api_e:
        logger.error(f"❌ facebook_business.api module error: {api_e}")
        
except ImportError as base_e:
    logger.error(f"❌ Facebook Business SDK base module not available: {base_e}")

# Import core components
try:
    from facebook_business.api import FacebookAdsApi
    from facebook_business.adobjects.adaccount import AdAccount
    from facebook_business.adobjects.page import Page
    logger.info("✅ Facebook Business SDK core components imported successfully")
except ImportError as e:
    logger.error(f"❌ Facebook Business SDK core import error: {e}")
    # Check if it's specifically a circular import
    if "circular import" in str(e).lower() or "partially initialized" in str(e).lower():
        logger.error("🔄 Circular import detected - this is a known issue with facebook-business SDK")

# Import exceptions separately to avoid circular import issues
try:
    from facebook_business.exceptions import FacebookRequestError
    logger.info("✅ Facebook Business SDK exceptions imported successfully")
    
    # Verify the exception class is properly formed
    if FacebookRequestError and isinstance(FacebookRequestError, type):
        is_exception = issubclass(FacebookRequestError, BaseException)
        logger.info(f"✅ FacebookRequestError is valid exception class: {is_exception}")
    else:
        logger.warning("⚠️ FacebookRequestError imported but not a proper class")
        
except ImportError as e:
    logger.error(f"❌ Facebook Business SDK exceptions import error: {e}")
    FacebookRequestError = None

logger = logging.getLogger('fb_client')

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
        # Debug: inspect FacebookRequestError import
        logger.info(f"DEBUG: FacebookRequestError imported as: {FacebookRequestError}")
        if FacebookRequestError is not None:
            try:
                is_exception = isinstance(FacebookRequestError, type) and issubclass(FacebookRequestError, BaseException)
                logger.info(f"DEBUG: Is subclass of BaseException? {is_exception}")
            except Exception as debug_e:
                logger.info(f"DEBUG: Error checking FacebookRequestError type: {debug_e}")
        else:
            logger.info("DEBUG: FacebookRequestError is None (import failed)")

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

        except Exception as e:
            # Log the actual exception type for debugging
            logger.error(f"❌ Error during Facebook API initialization: {e} (type: {type(e)})", exc_info=True)
            
            # Check if this is a Facebook SDK specific error
            if FacebookRequestError is not None and isinstance(e, FacebookRequestError):
                logger.error(f"❌ Facebook API request error during initialization: {e}")
            elif "facebook" in str(type(e)).lower():
                logger.error(f"❌ Facebook SDK related error: {e}")
            else:
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
        except Exception as e:
            # Log the actual exception type for debugging
            logger.error(f"Connection test failed: {e} (type: {type(e)})", exc_info=True)
            
            # Check if this is a Facebook SDK specific error
            if FacebookRequestError is not None and isinstance(e, FacebookRequestError):
                logger.error(f"Facebook API request error during connection test: {e}")
            elif "facebook" in str(type(e)).lower():
                logger.error(f"Facebook SDK related error during connection test: {e}")
            else:
                logger.error(f"Unexpected error during connection test: {e}")
            
            return {"success": False, "error": str(e)}

    def diagnose_sdk(self):
        """
        Diagnose Facebook SDK installation and imports.
        Returns: dict with diagnostic information
        """
        diagnostics = {
            "sdk_installed": False,
            "core_imports": {},
            "exception_imports": {},
            "version_info": None,
            "recommendations": []
        }

        # Check core imports
        diagnostics["core_imports"]["FacebookAdsApi"] = FacebookAdsApi is not None
        diagnostics["core_imports"]["AdAccount"] = AdAccount is not None
        diagnostics["core_imports"]["Page"] = Page is not None
        
        # Check exception imports
        diagnostics["exception_imports"]["FacebookRequestError"] = FacebookRequestError is not None
        
        if FacebookRequestError is not None:
            try:
                diagnostics["exception_imports"]["is_exception_subclass"] = issubclass(FacebookRequestError, BaseException)
            except Exception as e:
                diagnostics["exception_imports"]["subclass_check_error"] = str(e)

        # Check if SDK is generally available
        try:
            import facebook_business
            diagnostics["sdk_installed"] = True
            if hasattr(facebook_business, '__version__'):
                diagnostics["version_info"] = facebook_business.__version__
        except ImportError:
            diagnostics["recommendations"].append("Install facebook-business package: pip install facebook-business")

        # Add recommendations based on findings
        if not diagnostics["core_imports"]["FacebookAdsApi"]:
            diagnostics["recommendations"].append("Core Facebook SDK components not available - check installation")
        
        if not diagnostics["exception_imports"]["FacebookRequestError"]:
            diagnostics["recommendations"].append("Facebook SDK exceptions not available - use generic Exception handling")

        return diagnostics

# Global client instance
fb_client = FacebookClient()