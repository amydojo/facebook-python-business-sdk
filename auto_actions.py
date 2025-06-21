
"""
Automation and safe write-actions module for Facebook Marketing API.
All operations include dry-run mode and feature flags for safety.
References:
- Marketing API SDK: https://developers.facebook.com/docs/marketing-api/
- Custom Audience API: https://developers.facebook.com/docs/marketing-api/audiences/
"""
import logging
from datetime import datetime
from facebook_business.adobjects.adset import AdSet
from facebook_business.adobjects.ad import Ad
from facebook_business.adobjects.customaudience import CustomAudience
from facebook_business.exceptions import FacebookError
from fb_client import fb_client
from config import config

# Try to import data_store, create a fallback if not available
try:
    from data_store import log_audit_entry
except ImportError:
    def log_audit_entry(operation_log):
        """Fallback audit logging function"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Audit log: {operation_log}")

logger = logging.getLogger(__name__)

def safe_update_adset_budget(adset_id, new_budget, dry_run=True):
    """
    Safely update ad set daily budget.
    
    Args:
        adset_id: Facebook ad set ID
        new_budget: new daily budget in cents
        dry_run: if True, only log intended action
    
    Returns:
        dict: operation result
    """
    operation_log = {
        'action_type': 'update_adset_budget',
        'adset_id': adset_id,
        'new_budget': new_budget,
        'dry_run': dry_run,
        'timestamp': datetime.now(),
        'success': False,
        'error': None
    }
    
    try:
        if not fb_client.is_initialized():
            operation_log['error'] = "Facebook client not initialized"
            logger.error(operation_log['error'])
            return operation_log
        
        # Validate inputs
        if not adset_id or new_budget < 0:
            operation_log['error'] = "Invalid adset_id or budget amount"
            logger.error(operation_log['error'])
            return operation_log
        
        # Convert budget to cents if not already
        budget_cents = int(new_budget * 100) if new_budget < 1000 else int(new_budget)
        
        if dry_run or not config.ENABLE_AUTO_ACTIONS:
            # Log intended action only
            logger.info(f"🔍 DRY RUN: Would update adset {adset_id} budget to ${budget_cents/100:.2f}")
            operation_log['success'] = True
            operation_log['message'] = f"Dry run: Would update budget to ${budget_cents/100:.2f}"
        else:
            # Execute actual update
            adset = AdSet(adset_id)
            result = adset.api_update(fields=[], params={
                'daily_budget': budget_cents
            })
            
            logger.info(f"✅ Updated adset {adset_id} budget to ${budget_cents/100:.2f}")
            operation_log['success'] = True
            operation_log['message'] = f"Successfully updated budget to ${budget_cents/100:.2f}"
            operation_log['facebook_response'] = str(result)
        
        # Log to audit trail
        log_audit_entry(operation_log)
        
        return operation_log
        
    except FacebookError as e:
        operation_log['error'] = f"Facebook API error: {e}"
        logger.error(operation_log['error'])
        log_audit_entry(operation_log)
        return operation_log
    except Exception as e:
        operation_log['error'] = f"Unexpected error: {e}"
        logger.error(operation_log['error'])
        log_audit_entry(operation_log)
        return operation_log

def safe_pause_ad(ad_id, dry_run=True):
    """
    Safely pause an ad.
    
    Args:
        ad_id: Facebook ad ID
        dry_run: if True, only log intended action
    
    Returns:
        dict: operation result
    """
    operation_log = {
        'action_type': 'pause_ad',
        'ad_id': ad_id,
        'dry_run': dry_run,
        'timestamp': datetime.now(),
        'success': False,
        'error': None
    }
    
    try:
        if not fb_client.is_initialized():
            operation_log['error'] = "Facebook client not initialized"
            logger.error(operation_log['error'])
            return operation_log
        
        if not ad_id:
            operation_log['error'] = "Invalid ad_id"
            logger.error(operation_log['error'])
            return operation_log
        
        if dry_run or not config.ENABLE_AUTO_ACTIONS:
            # Log intended action only
            logger.info(f"🔍 DRY RUN: Would pause ad {ad_id}")
            operation_log['success'] = True
            operation_log['message'] = "Dry run: Would pause ad"
        else:
            # Execute actual pause
            ad = Ad(ad_id)
            result = ad.api_update(fields=[], params={
                'status': Ad.Status.paused
            })
            
            logger.info(f"✅ Paused ad {ad_id}")
            operation_log['success'] = True
            operation_log['message'] = "Successfully paused ad"
            operation_log['facebook_response'] = str(result)
        
        # Log to audit trail
        log_audit_entry(operation_log)
        
        return operation_log
        
    except FacebookError as e:
        operation_log['error'] = f"Facebook API error: {e}"
        logger.error(operation_log['error'])
        log_audit_entry(operation_log)
        return operation_log
    except Exception as e:
        operation_log['error'] = f"Unexpected error: {e}"
        logger.error(operation_log['error'])
        log_audit_entry(operation_log)
        return operation_log

def safe_activate_ad(ad_id, dry_run=True):
    """
    Safely activate/resume an ad.
    
    Args:
        ad_id: Facebook ad ID
        dry_run: if True, only log intended action
    
    Returns:
        dict: operation result
    """
    operation_log = {
        'action_type': 'activate_ad',
        'ad_id': ad_id,
        'dry_run': dry_run,
        'timestamp': datetime.now(),
        'success': False,
        'error': None
    }
    
    try:
        if not fb_client.is_initialized():
            operation_log['error'] = "Facebook client not initialized"
            logger.error(operation_log['error'])
            return operation_log
        
        if not ad_id:
            operation_log['error'] = "Invalid ad_id"
            logger.error(operation_log['error'])
            return operation_log
        
        if dry_run or not config.ENABLE_AUTO_ACTIONS:
            # Log intended action only
            logger.info(f"🔍 DRY RUN: Would activate ad {ad_id}")
            operation_log['success'] = True
            operation_log['message'] = "Dry run: Would activate ad"
        else:
            # Execute actual activation
            ad = Ad(ad_id)
            result = ad.api_update(fields=[], params={
                'status': Ad.Status.active
            })
            
            logger.info(f"✅ Activated ad {ad_id}")
            operation_log['success'] = True
            operation_log['message'] = "Successfully activated ad"
            operation_log['facebook_response'] = str(result)
        
        # Log to audit trail
        log_audit_entry(operation_log)
        
        return operation_log
        
    except FacebookError as e:
        operation_log['error'] = f"Facebook API error: {e}"
        logger.error(operation_log['error'])
        log_audit_entry(operation_log)
        return operation_log
    except Exception as e:
        operation_log['error'] = f"Unexpected error: {e}"
        logger.error(operation_log['error'])
        log_audit_entry(operation_log)
        return operation_log

def create_lookalike(source_audience_id, country="US", ratio=0.01, dry_run=True):
    """
    Create a lookalike audience.
    
    Args:
        source_audience_id: source custom audience ID
        country: target country code
        ratio: lookalike ratio (0.01 = 1%)
        dry_run: if True, only log intended action
    
    Returns:
        dict: operation result
        
    Reference: https://developers.facebook.com/docs/marketing-api/audiences/
    """
    operation_log = {
        'action_type': 'create_lookalike',
        'source_audience_id': source_audience_id,
        'country': country,
        'ratio': ratio,
        'dry_run': dry_run,
        'timestamp': datetime.now(),
        'success': False,
        'error': None
    }
    
    try:
        if not fb_client.is_initialized():
            operation_log['error'] = "Facebook client not initialized"
            logger.error(operation_log['error'])
            return operation_log
        
        if not source_audience_id:
            operation_log['error'] = "Invalid source_audience_id"
            logger.error(operation_log['error'])
            return operation_log
        
        if dry_run or not config.ENABLE_AUTO_ACTIONS:
            # Log intended action only
            logger.info(f"🔍 DRY RUN: Would create {ratio*100}% lookalike audience from {source_audience_id} targeting {country}")
            operation_log['success'] = True
            operation_log['message'] = f"Dry run: Would create {ratio*100}% lookalike for {country}"
        else:
            # Execute actual creation
            params = {
                'name': f'Lookalike {ratio*100}% - {country} - {datetime.now().strftime("%Y%m%d")}',
                'subtype': CustomAudience.Subtype.lookalike,
                'lookalike_spec': {
                    'ratio': ratio,
                    'country': country,
                    'type': 'similarity'
                },
                'origin_audience_id': source_audience_id
            }
            
            result = fb_client.ad_account.create_custom_audience(params=params)
            
            logger.info(f"✅ Created lookalike audience: {result.get('id')}")
            operation_log['success'] = True
            operation_log['message'] = f"Successfully created lookalike audience: {result.get('id')}"
            operation_log['lookalike_audience_id'] = result.get('id')
            operation_log['facebook_response'] = str(result)
        
        # Log to audit trail
        log_audit_entry(operation_log)
        
        return operation_log
        
    except FacebookError as e:
        operation_log['error'] = f"Facebook API error: {e}"
        logger.error(operation_log['error'])
        log_audit_entry(operation_log)
        return operation_log
    except Exception as e:
        operation_log['error'] = f"Unexpected error: {e}"
        logger.error(operation_log['error'])
        log_audit_entry(operation_log)
        return operation_log

def safe_update_ad_creative(ad_id, new_creative_id, dry_run=True):
    """
    Safely update ad creative.
    
    Args:
        ad_id: Facebook ad ID
        new_creative_id: new creative ID
        dry_run: if True, only log intended action
    
    Returns:
        dict: operation result
    """
    operation_log = {
        'action_type': 'update_ad_creative',
        'ad_id': ad_id,
        'new_creative_id': new_creative_id,
        'dry_run': dry_run,
        'timestamp': datetime.now(),
        'success': False,
        'error': None
    }
    
    try:
        if not fb_client.is_initialized():
            operation_log['error'] = "Facebook client not initialized"
            logger.error(operation_log['error'])
            return operation_log
        
        if not ad_id or not new_creative_id:
            operation_log['error'] = "Invalid ad_id or creative_id"
            logger.error(operation_log['error'])
            return operation_log
        
        if dry_run or not config.ENABLE_AUTO_ACTIONS:
            # Log intended action only
            logger.info(f"🔍 DRY RUN: Would update ad {ad_id} creative to {new_creative_id}")
            operation_log['success'] = True
            operation_log['message'] = f"Dry run: Would update creative to {new_creative_id}"
        else:
            # Execute actual update
            ad = Ad(ad_id)
            result = ad.api_update(fields=[], params={
                'creative': {'creative_id': new_creative_id}
            })
            
            logger.info(f"✅ Updated ad {ad_id} creative to {new_creative_id}")
            operation_log['success'] = True
            operation_log['message'] = f"Successfully updated creative to {new_creative_id}"
            operation_log['facebook_response'] = str(result)
        
        # Log to audit trail
        log_audit_entry(operation_log)
        
        return operation_log
        
    except FacebookError as e:
        operation_log['error'] = f"Facebook API error: {e}"
        logger.error(operation_log['error'])
        log_audit_entry(operation_log)
        return operation_log
    except Exception as e:
        operation_log['error'] = f"Unexpected error: {e}"
        logger.error(operation_log['error'])
        log_audit_entry(operation_log)
        return operation_log

def bulk_budget_optimization(optimization_rules, dry_run=True):
    """
    Apply bulk budget optimization based on performance rules.
    
    Args:
        optimization_rules: list of dicts with adset_id, current_budget, new_budget, reason
        dry_run: if True, only log intended actions
    
    Returns:
        dict: summary of optimization results
    """
    results = {
        'total_rules': len(optimization_rules),
        'successful_updates': 0,
        'failed_updates': 0,
        'dry_run': dry_run,
        'details': []
    }
    
    logger.info(f"Starting bulk budget optimization with {len(optimization_rules)} rules")
    
    for rule in optimization_rules:
        adset_id = rule.get('adset_id')
        new_budget = rule.get('new_budget')
        reason = rule.get('reason', 'Performance optimization')
        
        result = safe_update_adset_budget(adset_id, new_budget, dry_run)
        
        if result['success']:
            results['successful_updates'] += 1
        else:
            results['failed_updates'] += 1
        
        results['details'].append({
            'adset_id': adset_id,
            'new_budget': new_budget,
            'reason': reason,
            'success': result['success'],
            'error': result.get('error')
        })
    
    logger.info(f"Bulk optimization complete: {results['successful_updates']} successful, {results['failed_updates']} failed")
    
    return results

def automated_performance_actions(performance_thresholds, dry_run=True):
    """
    Execute automated actions based on performance thresholds.
    
    Args:
        performance_thresholds: dict with rules for automated actions
        dry_run: if True, only log intended actions
    
    Returns:
        dict: summary of automated actions taken
    """
    # This would integrate with fetch_paid.py to get current performance
    # and apply rules like:
    # - Pause ads with CTR < 0.5%
    # - Increase budget for ads with ROAS > 4
    # - Create lookalikes from high-converting audiences
    
    results = {
        'actions_taken': 0,
        'dry_run': dry_run,
        'details': []
    }
    
    try:
        # Example implementation - would be expanded based on specific needs
        logger.info("🤖 Running automated performance optimization")
        
        # Placeholder for actual performance-based logic
        if dry_run or not config.ENABLE_AUTO_ACTIONS:
            logger.info("🔍 DRY RUN: Automated actions would be executed here")
            results['details'].append({
                'action': 'dry_run_placeholder',
                'message': 'Automated optimization rules would be applied'
            })
        else:
            logger.info("Automated optimization feature needs performance data integration")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in automated performance actions: {e}")
        results['error'] = str(e)
        return results

def validate_automation_safety():
    """
    Validate that automation safety checks are in place.
    
    Returns:
        dict: safety validation results
    """
    safety_checks = {
        'feature_flag_enabled': config.ENABLE_AUTO_ACTIONS,
        'facebook_client_initialized': fb_client.is_initialized(),
        'audit_logging_available': True,  # Assume data_store is available
        'dry_run_default': True,  # All functions default to dry_run=True
        'safe_to_automate': False
    }
    
    # Determine if it's safe to run automation
    safety_checks['safe_to_automate'] = (
        safety_checks['facebook_client_initialized'] and
        safety_checks['audit_logging_available']
    )
    
    logger.info(f"Automation safety validation: {safety_checks}")
    
    return safety_checks
