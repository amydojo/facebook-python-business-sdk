
"""
Advanced Analytics Module for Campaign Performance and Optimization
Provides deep insights, forecasting, and optimization recommendations
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class CampaignAnalyzer:
    """Advanced campaign performance analyzer"""
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
    
    def analyze_roi_performance(self, campaign_data: pd.DataFrame, conversion_value: float = 50.0) -> Dict[str, Any]:
        """
        Comprehensive ROI analysis for campaigns
        
        Args:
            campaign_data: DataFrame with campaign performance data
            conversion_value: Average value per conversion
            
        Returns:
            Dict with ROI analysis results
        """
        try:
            if campaign_data.empty:
                return {'error': 'No campaign data provided'}
            
            # Calculate ROI metrics
            analysis = {}
            
            # Ensure numeric columns
            numeric_cols = ['spend', 'impressions', 'clicks', 'conversions']
            for col in numeric_cols:
                if col in campaign_data.columns:
                    campaign_data[col] = pd.to_numeric(campaign_data[col], errors='coerce').fillna(0)
            
            # Calculate derived metrics
            campaign_data['revenue'] = campaign_data.get('conversions', 0) * conversion_value
            campaign_data['roi'] = np.where(
                campaign_data['spend'] > 0,
                (campaign_data['revenue'] - campaign_data['spend']) / campaign_data['spend'] * 100,
                0
            )
            campaign_data['ctr'] = np.where(
                campaign_data['impressions'] > 0,
                campaign_data['clicks'] / campaign_data['impressions'] * 100,
                0
            )
            campaign_data['conversion_rate'] = np.where(
                campaign_data['clicks'] > 0,
                campaign_data.get('conversions', 0) / campaign_data['clicks'] * 100,
                0
            )
            
            # Summary statistics
            analysis['summary'] = {
                'total_spend': campaign_data['spend'].sum(),
                'total_revenue': campaign_data['revenue'].sum(),
                'total_roi': (campaign_data['revenue'].sum() - campaign_data['spend'].sum()) / campaign_data['spend'].sum() * 100 if campaign_data['spend'].sum() > 0 else 0,
                'avg_ctr': campaign_data['ctr'].mean(),
                'avg_conversion_rate': campaign_data['conversion_rate'].mean(),
                'campaign_count': len(campaign_data)
            }
            
            # Top performers
            analysis['top_performers'] = {
                'highest_roi': campaign_data.nlargest(5, 'roi')[['campaign_name', 'roi', 'spend', 'revenue']].to_dict('records'),
                'highest_ctr': campaign_data.nlargest(5, 'ctr')[['campaign_name', 'ctr', 'impressions', 'clicks']].to_dict('records'),
                'most_conversions': campaign_data.nlargest(5, 'conversions')[['campaign_name', 'conversions', 'spend']].to_dict('records')
            }
            
            # Underperformers
            analysis['underperformers'] = {
                'lowest_roi': campaign_data.nsmallest(3, 'roi')[['campaign_name', 'roi', 'spend']].to_dict('records'),
                'lowest_ctr': campaign_data.nsmallest(3, 'ctr')[['campaign_name', 'ctr']].to_dict('records')
            }
            
            # Optimization recommendations
            analysis['recommendations'] = self._generate_roi_recommendations(campaign_data)
            
            return analysis
            
        except Exception as e:
            logger.error(f"ROI analysis failed: {e}")
            return {'error': str(e)}
    
    def analyze_creative_performance(self, campaign_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze creative asset performance across campaigns
        """
        try:
            if 'creative_name' not in campaign_data.columns:
                return {'error': 'Creative data not available'}
            
            # Group by creative
            creative_performance = campaign_data.groupby('creative_name').agg({
                'impressions': 'sum',
                'clicks': 'sum',
                'spend': 'sum',
                'conversions': 'sum' if 'conversions' in campaign_data.columns else lambda x: 0
            }).reset_index()
            
            # Calculate creative metrics
            creative_performance['ctr'] = np.where(
                creative_performance['impressions'] > 0,
                creative_performance['clicks'] / creative_performance['impressions'] * 100,
                0
            )
            creative_performance['cpm'] = np.where(
                creative_performance['impressions'] > 0,
                creative_performance['spend'] / creative_performance['impressions'] * 1000,
                0
            )
            creative_performance['cpc'] = np.where(
                creative_performance['clicks'] > 0,
                creative_performance['spend'] / creative_performance['clicks'],
                0
            )
            
            analysis = {
                'performance_table': creative_performance.to_dict('records'),
                'top_creatives': {
                    'highest_ctr': creative_performance.nlargest(3, 'ctr')[['creative_name', 'ctr']].to_dict('records'),
                    'lowest_cpm': creative_performance.nsmallest(3, 'cpm')[['creative_name', 'cpm']].to_dict('records'),
                    'most_efficient': creative_performance.nsmallest(3, 'cpc')[['creative_name', 'cpc']].to_dict('records')
                }
            }
            
            # Creative fatigue detection
            analysis['fatigue_analysis'] = self._detect_creative_fatigue(campaign_data)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Creative analysis failed: {e}")
            return {'error': str(e)}
    
    def forecast_performance(self, historical_data: pd.DataFrame, days_ahead: int = 30) -> Dict[str, Any]:
        """
        Forecast campaign performance using machine learning
        """
        try:
            if len(historical_data) < 7:
                return {'error': 'Insufficient historical data for forecasting'}
            
            # Prepare features
            features = self._prepare_forecast_features(historical_data)
            
            if features.empty:
                return {'error': 'Could not prepare features for forecasting'}
            
            # Train models for key metrics
            forecasts = {}
            metrics_to_forecast = ['spend', 'impressions', 'clicks']
            
            for metric in metrics_to_forecast:
                if metric in features.columns:
                    try:
                        forecast = self._train_and_forecast(features, metric, days_ahead)
                        forecasts[metric] = forecast
                    except Exception as e:
                        logger.warning(f"Could not forecast {metric}: {e}")
            
            # Calculate derived forecasts
            if 'spend' in forecasts and 'clicks' in forecasts:
                forecasts['cpc'] = [
                    s/c if c > 0 else 0 
                    for s, c in zip(forecasts['spend'], forecasts['clicks'])
                ]
            
            return {
                'forecasts': forecasts,
                'confidence': 'Medium',  # Would be calculated from model metrics
                'recommendations': self._generate_forecast_recommendations(forecasts)
            }
            
        except Exception as e:
            logger.error(f"Performance forecasting failed: {e}")
            return {'error': str(e)}
    
    def optimize_budget_allocation(self, campaign_data: pd.DataFrame, total_budget: float) -> Dict[str, Any]:
        """
        Optimize budget allocation across campaigns based on performance
        """
        try:
            if campaign_data.empty or total_budget <= 0:
                return {'error': 'Invalid input for budget optimization'}
            
            # Calculate efficiency scores
            campaign_data['efficiency_score'] = self._calculate_efficiency_score(campaign_data)
            
            # Allocate budget based on efficiency
            total_efficiency = campaign_data['efficiency_score'].sum()
            
            if total_efficiency > 0:
                campaign_data['recommended_budget'] = (
                    campaign_data['efficiency_score'] / total_efficiency * total_budget
                )
            else:
                # Equal allocation if no efficiency data
                campaign_data['recommended_budget'] = total_budget / len(campaign_data)
            
            # Generate allocation recommendations
            allocation = {
                'budget_allocation': campaign_data[['campaign_name', 'efficiency_score', 'recommended_budget']].to_dict('records'),
                'reallocation_impact': self._calculate_reallocation_impact(campaign_data),
                'optimization_summary': {
                    'total_budget': total_budget,
                    'high_efficiency_campaigns': len(campaign_data[campaign_data['efficiency_score'] > campaign_data['efficiency_score'].median()]),
                    'recommended_shifts': self._identify_budget_shifts(campaign_data)
                }
            }
            
            return allocation
            
        except Exception as e:
            logger.error(f"Budget optimization failed: {e}")
            return {'error': str(e)}
    
    def _generate_roi_recommendations(self, campaign_data: pd.DataFrame) -> List[str]:
        """Generate ROI-based optimization recommendations"""
        recommendations = []
        
        # High spend, low ROI campaigns
        low_roi_high_spend = campaign_data[
            (campaign_data['roi'] < 0) & 
            (campaign_data['spend'] > campaign_data['spend'].median())
        ]
        
        if not low_roi_high_spend.empty:
            recommendations.append(f"⚠️ {len(low_roi_high_spend)} high-spend campaigns have negative ROI - consider pausing or optimizing")
        
        # High CTR, low conversion rate
        high_ctr_low_conv = campaign_data[
            (campaign_data['ctr'] > campaign_data['ctr'].median()) &
            (campaign_data['conversion_rate'] < campaign_data['conversion_rate'].median())
        ]
        
        if not high_ctr_low_conv.empty:
            recommendations.append(f"🎯 {len(high_ctr_low_conv)} campaigns have good CTR but low conversions - optimize landing pages")
        
        # Budget reallocation opportunities
        top_roi = campaign_data.nlargest(3, 'roi')
        if not top_roi.empty:
            recommendations.append(f"💰 Consider increasing budget for top ROI campaigns: {', '.join(top_roi['campaign_name'].head(2))}")
        
        return recommendations
    
    def _detect_creative_fatigue(self, campaign_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect creative fatigue patterns"""
        # Simplified fatigue detection
        fatigue_indicators = {}
        
        if 'creative_name' in campaign_data.columns:
            creative_freq = campaign_data['creative_name'].value_counts()
            overused_creatives = creative_freq[creative_freq > 5].index.tolist()
            
            fatigue_indicators = {
                'overused_creatives': overused_creatives,
                'recommendation': "Consider refreshing creatives used in more than 5 campaigns"
            }
        
        return fatigue_indicators
    
    def _prepare_forecast_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for forecasting model"""
        try:
            # Create time-based features
            features = data.copy()
            
            if 'date_start' in features.columns:
                features['date'] = pd.to_datetime(features['date_start'])
            elif 'timestamp' in features.columns:
                features['date'] = pd.to_datetime(features['timestamp'])
            else:
                # Create synthetic date range
                features['date'] = pd.date_range(start='2024-01-01', periods=len(features), freq='D')
            
            # Sort by date
            features = features.sort_values('date')
            
            # Create lag features
            numeric_cols = ['spend', 'impressions', 'clicks']
            for col in numeric_cols:
                if col in features.columns:
                    features[f'{col}_lag1'] = features[col].shift(1)
                    features[f'{col}_lag7'] = features[col].shift(7)
            
            # Create trend features
            features['day_of_week'] = features['date'].dt.dayofweek
            features['day_of_month'] = features['date'].dt.day
            
            return features.dropna()
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return pd.DataFrame()
    
    def _train_and_forecast(self, features: pd.DataFrame, target_metric: str, days_ahead: int) -> List[float]:
        """Train ML model and generate forecast"""
        try:
            # Prepare training data
            feature_cols = [col for col in features.columns if col.endswith('_lag1') or col.endswith('_lag7') or col in ['day_of_week', 'day_of_month']]
            
            X = features[feature_cols].fillna(0)
            y = features[target_metric].fillna(0)
            
            if len(X) < 5:  # Need minimum data
                return [y.mean()] * days_ahead
            
            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # Generate forecast
            last_values = X.iloc[-1:].values
            forecasts = []
            
            for _ in range(days_ahead):
                pred = model.predict(last_values)[0]
                forecasts.append(max(0, pred))  # Ensure non-negative
                
                # Update features for next prediction (simplified)
                last_values[0][0] = pred  # Update lag1 feature
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Forecasting failed for {target_metric}: {e}")
            return [0] * days_ahead
    
    def _calculate_efficiency_score(self, campaign_data: pd.DataFrame) -> pd.Series:
        """Calculate campaign efficiency score for budget allocation"""
        try:
            # Normalize metrics to 0-1 scale
            roi_norm = (campaign_data['roi'] - campaign_data['roi'].min()) / (campaign_data['roi'].max() - campaign_data['roi'].min() + 1e-8)
            ctr_norm = (campaign_data['ctr'] - campaign_data['ctr'].min()) / (campaign_data['ctr'].max() - campaign_data['ctr'].min() + 1e-8)
            
            # Weighted efficiency score
            efficiency = (roi_norm * 0.6) + (ctr_norm * 0.4)
            
            return efficiency.fillna(0)
            
        except Exception as e:
            logger.error(f"Efficiency calculation failed: {e}")
            return pd.Series([0] * len(campaign_data))
    
    def _calculate_reallocation_impact(self, campaign_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate expected impact of budget reallocation"""
        try:
            current_total_roi = (campaign_data['revenue'].sum() - campaign_data['spend'].sum()) / campaign_data['spend'].sum() * 100
            
            # Simulate optimal allocation impact
            high_efficiency = campaign_data[campaign_data['efficiency_score'] > campaign_data['efficiency_score'].median()]
            
            if not high_efficiency.empty:
                projected_roi_improvement = high_efficiency['roi'].mean() - campaign_data['roi'].mean()
            else:
                projected_roi_improvement = 0
            
            return {
                'current_roi': current_total_roi,
                'projected_roi_improvement': projected_roi_improvement,
                'potential_additional_revenue': projected_roi_improvement * campaign_data['spend'].sum() / 100
            }
            
        except Exception as e:
            logger.error(f"Reallocation impact calculation failed: {e}")
            return {'current_roi': 0, 'projected_roi_improvement': 0, 'potential_additional_revenue': 0}
    
    def _identify_budget_shifts(self, campaign_data: pd.DataFrame) -> List[str]:
        """Identify specific budget shift recommendations"""
        recommendations = []
        
        try:
            # High efficiency campaigns that could use more budget
            high_eff_low_budget = campaign_data[
                (campaign_data['efficiency_score'] > campaign_data['efficiency_score'].quantile(0.7)) &
                (campaign_data['spend'] < campaign_data['spend'].median())
            ]
            
            # Low efficiency campaigns that should reduce budget
            low_eff_high_budget = campaign_data[
                (campaign_data['efficiency_score'] < campaign_data['efficiency_score'].quantile(0.3)) &
                (campaign_data['spend'] > campaign_data['spend'].median())
            ]
            
            if not high_eff_low_budget.empty:
                recommendations.append(f"Increase budget for {len(high_eff_low_budget)} high-efficiency campaigns")
            
            if not low_eff_high_budget.empty:
                recommendations.append(f"Reduce budget for {len(low_eff_high_budget)} low-efficiency campaigns")
            
        except Exception as e:
            logger.error(f"Budget shift identification failed: {e}")
        
        return recommendations
    
    def _generate_forecast_recommendations(self, forecasts: Dict[str, List[float]]) -> List[str]:
        """Generate recommendations based on forecasts"""
        recommendations = []
        
        try:
            if 'spend' in forecasts:
                total_forecasted_spend = sum(forecasts['spend'])
                recommendations.append(f"Forecasted 30-day spend: ${total_forecasted_spend:.2f}")
                
                # Trend analysis
                if len(forecasts['spend']) >= 7:
                    week1_avg = np.mean(forecasts['spend'][:7])
                    week4_avg = np.mean(forecasts['spend'][-7:])
                    
                    if week4_avg > week1_avg * 1.1:
                        recommendations.append("⚠️ Spend trending upward - monitor budget closely")
                    elif week4_avg < week1_avg * 0.9:
                        recommendations.append("📉 Spend trending downward - consider budget optimization")
            
            if 'cpc' in forecasts:
                avg_cpc = np.mean(forecasts['cpc'])
                if avg_cpc > 2.0:
                    recommendations.append("💰 High CPC forecasted - optimize targeting and creatives")
                elif avg_cpc < 0.5:
                    recommendations.append("✅ Low CPC forecasted - good opportunity to scale")
        
        except Exception as e:
            logger.error(f"Forecast recommendations failed: {e}")
        
        return recommendations

def create_performance_dashboard_data(campaign_data: pd.DataFrame, organic_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Create comprehensive dashboard data combining paid and organic performance
    """
    try:
        dashboard_data = {
            'summary': {},
            'trends': {},
            'recommendations': [],
            'alerts': []
        }
        
        # Paid campaign summary
        if not campaign_data.empty:
            paid_summary = {
                'total_spend': pd.to_numeric(campaign_data['spend'], errors='coerce').sum(),
                'total_impressions': pd.to_numeric(campaign_data['impressions'], errors='coerce').sum(),
                'total_clicks': pd.to_numeric(campaign_data['clicks'], errors='coerce').sum(),
                'avg_ctr': pd.to_numeric(campaign_data.get('ctr', 0), errors='coerce').mean(),
                'campaign_count': len(campaign_data)
            }
            dashboard_data['summary']['paid'] = paid_summary
        
        # Organic summary
        if not organic_data.empty:
            organic_reach = organic_data[organic_data['metric'] == 'reach']['value'].sum() if 'reach' in organic_data['metric'].values else 0
            organic_engagement = organic_data[organic_data['metric'] == 'likes']['value'].sum() if 'likes' in organic_data['metric'].values else 0
            
            organic_summary = {
                'total_reach': organic_reach,
                'total_engagement': organic_engagement,
                'engagement_rate': (organic_engagement / organic_reach * 100) if organic_reach > 0 else 0,
                'post_count': len(organic_data['media_id'].unique()) if 'media_id' in organic_data.columns else 0
            }
            dashboard_data['summary']['organic'] = organic_summary
        
        # Cross-platform insights
        if not campaign_data.empty and not organic_data.empty:
            # Cost efficiency comparison
            paid_cost_per_impression = paid_summary['total_spend'] / paid_summary['total_impressions'] * 1000 if paid_summary['total_impressions'] > 0 else 0
            organic_cpm_equivalent = 0  # Organic is free
            
            dashboard_data['summary']['comparison'] = {
                'paid_cpm': paid_cost_per_impression,
                'organic_cpm_equivalent': organic_cpm_equivalent,
                'efficiency_ratio': 'Organic wins on cost, Paid wins on scale'
            }
        
        # Performance alerts
        if not campaign_data.empty:
            high_spend_low_performance = campaign_data[
                (pd.to_numeric(campaign_data['spend'], errors='coerce') > campaign_data['spend'].astype(float).median()) &
                (pd.to_numeric(campaign_data.get('ctr', 0), errors='coerce') < 1.0)
            ]
            
            if not high_spend_low_performance.empty:
                dashboard_data['alerts'].append(f"⚠️ {len(high_spend_low_performance)} high-spend campaigns have low CTR")
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Dashboard data creation failed: {e}")
        return {'error': str(e)}

# Performance optimization utilities
def identify_optimization_opportunities(campaign_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Identify specific optimization opportunities"""
    opportunities = []
    
    try:
        if campaign_data.empty:
            return opportunities
        
        # Ensure numeric data
        numeric_cols = ['spend', 'impressions', 'clicks', 'ctr']
        for col in numeric_cols:
            if col in campaign_data.columns:
                campaign_data[col] = pd.to_numeric(campaign_data[col], errors='coerce').fillna(0)
        
        # High impression, low click campaigns
        low_ctr_campaigns = campaign_data[
            (campaign_data['impressions'] > campaign_data['impressions'].median()) &
            (campaign_data['ctr'] < 1.0)
        ]
        
        if not low_ctr_campaigns.empty:
            opportunities.append({
                'type': 'creative_optimization',
                'priority': 'high',
                'description': f'{len(low_ctr_campaigns)} campaigns need creative refresh',
                'campaigns': low_ctr_campaigns['campaign_name'].tolist()[:3],
                'action': 'Test new ad creatives and copy'
            })
        
        # High click, low conversion campaigns (if conversion data available)
        if 'conversions' in campaign_data.columns:
            campaign_data['conversion_rate'] = np.where(
                campaign_data['clicks'] > 0,
                pd.to_numeric(campaign_data['conversions'], errors='coerce') / campaign_data['clicks'] * 100,
                0
            )
            
            low_conversion_campaigns = campaign_data[
                (campaign_data['clicks'] > campaign_data['clicks'].median()) &
                (campaign_data['conversion_rate'] < 2.0)
            ]
            
            if not low_conversion_campaigns.empty:
                opportunities.append({
                    'type': 'landing_page_optimization',
                    'priority': 'high',
                    'description': f'{len(low_conversion_campaigns)} campaigns need landing page optimization',
                    'campaigns': low_conversion_campaigns['campaign_name'].tolist()[:3],
                    'action': 'Optimize landing pages and conversion flow'
                })
        
        # Budget reallocation opportunities
        if len(campaign_data) > 1:
            # High performing campaigns with low spend
            top_performers = campaign_data.nlargest(3, 'ctr')
            low_spend_high_performance = top_performers[
                top_performers['spend'] < campaign_data['spend'].median()
            ]
            
            if not low_spend_high_performance.empty:
                opportunities.append({
                    'type': 'budget_reallocation',
                    'priority': 'medium',
                    'description': f'{len(low_spend_high_performance)} high-performing campaigns could use more budget',
                    'campaigns': low_spend_high_performance['campaign_name'].tolist(),
                    'action': 'Increase budget allocation for top performers'
                })
        
    except Exception as e:
        logger.error(f"Opportunity identification failed: {e}")
    
    return opportunities

def generate_executive_summary(campaign_data: pd.DataFrame, organic_data: pd.DataFrame) -> Dict[str, Any]:
    """Generate executive summary of all marketing performance"""
    try:
        summary = {
            'overview': {},
            'key_metrics': {},
            'top_insights': [],
            'action_items': []
        }
        
        # Overall performance overview
        total_paid_spend = pd.to_numeric(campaign_data['spend'], errors='coerce').sum() if not campaign_data.empty else 0
        total_organic_reach = organic_data[organic_data['metric'] == 'reach']['value'].sum() if not organic_data.empty and 'reach' in organic_data['metric'].values else 0
        
        summary['overview'] = {
            'reporting_period': '7 days',
            'total_marketing_spend': total_paid_spend,
            'total_organic_reach': total_organic_reach,
            'active_campaigns': len(campaign_data) if not campaign_data.empty else 0,
            'organic_posts': len(organic_data['media_id'].unique()) if not organic_data.empty and 'media_id' in organic_data.columns else 0
        }
        
        # Key performance insights
        if not campaign_data.empty:
            best_campaign = campaign_data.loc[campaign_data['ctr'].astype(float).idxmax()] if 'ctr' in campaign_data.columns else None
            if best_campaign is not None:
                summary['top_insights'].append(f"🏆 Best performing campaign: {best_campaign.get('campaign_name', 'Unknown')} with {best_campaign.get('ctr', 0):.2f}% CTR")
        
        if not organic_data.empty:
            top_organic_post = organic_data.loc[organic_data['value'].idxmax()] if 'value' in organic_data.columns else None
            if top_organic_post is not None:
                summary['top_insights'].append(f"📸 Top organic post reached {top_organic_post.get('value', 0):,} people")
        
        # Action items
        if total_paid_spend > 1000:
            summary['action_items'].append("💰 High ad spend detected - review ROI and budget allocation")
        
        if total_organic_reach > 10000:
            summary['action_items'].append("📈 Strong organic reach - consider promoting top content with paid ads")
        
        return summary
        
    except Exception as e:
        logger.error(f"Executive summary generation failed: {e}")
        return {'error': str(e)}
