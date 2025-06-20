
"""
Forecast and what-if analysis module for budget optimization.
Uses scikit-learn for simple regression models and Prophet for time-series forecasting.
"""
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available - time series forecasting will be limited")

from openai_client import call_chat

logger = logging.getLogger(__name__)

def train_forecast_model(df_spend_rev, model_type='linear'):
    """
    Train forecast models for revenue prediction based on spend.
    
    Args:
        df_spend_rev: DataFrame with columns campaign_id, spend, revenue, date (optional)
        model_type: 'linear' for LinearRegression, 'prophet' for time-series
    
    Returns:
        dict: {campaign_id: trained_model}
    """
    if df_spend_rev.empty:
        logger.warning("Empty spend/revenue data provided")
        return {}
    
    # Validate required columns
    required_cols = ['campaign_id', 'spend', 'revenue']
    missing_cols = [col for col in required_cols if col not in df_spend_rev.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return {}
    
    models = {}
    
    try:
        for campaign_id, campaign_data in df_spend_rev.groupby('campaign_id'):
            if len(campaign_data) < 3:  # Need minimum data points
                logger.warning(f"Insufficient data for campaign {campaign_id} ({len(campaign_data)} points)")
                continue
            
            if model_type == 'linear':
                model = _train_linear_model(campaign_data)
            elif model_type == 'prophet' and PROPHET_AVAILABLE:
                model = _train_prophet_model(campaign_data)
            else:
                logger.warning(f"Unknown model type {model_type} or Prophet not available")
                model = _train_linear_model(campaign_data)
            
            if model is not None:
                models[campaign_id] = model
                logger.info(f"Trained {model_type} model for campaign {campaign_id}")
        
        logger.info(f"Successfully trained models for {len(models)} campaigns")
        return models
        
    except Exception as e:
        logger.error(f"Error training forecast models: {e}")
        return {}

def _train_linear_model(campaign_data):
    """Train a linear regression model for a single campaign."""
    try:
        # Prepare data
        X = campaign_data[['spend']].values
        y = campaign_data['revenue'].values
        
        # Remove rows with zero spend or negative values
        valid_indices = (X.flatten() > 0) & (y >= 0)
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) < 2:
            return None
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate metrics
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Store model with metadata
        model_info = {
            'model': model,
            'type': 'linear',
            'mae': mae,
            'r2': r2,
            'data_points': len(X),
            'avg_spend': np.mean(X),
            'avg_revenue': np.mean(y)
        }
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error training linear model: {e}")
        return None

def _train_prophet_model(campaign_data):
    """Train a Prophet time-series model for a single campaign."""
    if not PROPHET_AVAILABLE:
        return None
    
    try:
        # Prophet requires 'ds' (date) and 'y' (target) columns
        if 'date' not in campaign_data.columns:
            logger.warning("Prophet model requires 'date' column")
            return None
        
        # Prepare data for Prophet
        prophet_data = campaign_data[['date', 'revenue']].copy()
        prophet_data.columns = ['ds', 'y']
        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
        
        # Add spend as additional regressor
        prophet_data['spend'] = campaign_data['spend'].values
        
        # Initialize and train Prophet model
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        model.add_regressor('spend')
        model.fit(prophet_data)
        
        # Calculate in-sample metrics
        forecast = model.predict(prophet_data)
        mae = mean_absolute_error(prophet_data['y'], forecast['yhat'])
        
        model_info = {
            'model': model,
            'type': 'prophet',
            'mae': mae,
            'data_points': len(prophet_data),
            'date_range': (prophet_data['ds'].min(), prophet_data['ds'].max())
        }
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error training Prophet model: {e}")
        return None

def forecast_revenue(models, scenario):
    """
    Forecast revenue for given spend scenario.
    
    Args:
        models: dict from train_forecast_model()
        scenario: dict {campaign_id: new_spend_amount}
    
    Returns:
        dict {campaign_id: predicted_revenue}
    """
    if not models or not scenario:
        logger.warning("Empty models or scenario provided")
        return {}
    
    predictions = {}
    
    try:
        for campaign_id, new_spend in scenario.items():
            if campaign_id not in models:
                logger.warning(f"No model available for campaign {campaign_id}")
                continue
            
            model_info = models[campaign_id]
            model = model_info['model']
            model_type = model_info['type']
            
            if model_type == 'linear':
                # Predict with linear model
                pred_revenue = model.predict([[new_spend]])[0]
                predictions[campaign_id] = max(0, pred_revenue)  # Ensure non-negative
                
            elif model_type == 'prophet':
                # Predict with Prophet model
                # Create future dataframe
                future_df = pd.DataFrame({
                    'ds': [pd.Timestamp.now()],
                    'spend': [new_spend]
                })
                forecast = model.predict(future_df)
                pred_revenue = forecast['yhat'].iloc[0]
                predictions[campaign_id] = max(0, pred_revenue)
            
            logger.info(f"Predicted revenue for campaign {campaign_id}: ${pred_revenue:.2f}")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error forecasting revenue: {e}")
        return {}

def summarize_forecast(scenario_preds, baseline_period_data=None):
    """
    Generate AI-powered summary of forecast results.
    
    Args:
        scenario_preds: dict from forecast_revenue()
        baseline_period_data: dict with baseline metrics for comparison
    
    Returns:
        str: AI-generated summary and recommendations
    """
    try:
        # Prepare context for AI summary
        total_predicted = sum(scenario_preds.values())
        num_campaigns = len(scenario_preds)
        
        context = {
            "total_predicted_revenue": total_predicted,
            "num_campaigns": num_campaigns,
            "campaign_predictions": scenario_preds
        }
        
        if baseline_period_data:
            baseline_total = baseline_period_data.get('total_revenue', 0)
            context["baseline_total_revenue"] = baseline_total
            context["predicted_vs_baseline"] = (total_predicted - baseline_total) / baseline_total * 100 if baseline_total > 0 else 0
        
        # Create prompt for AI summary
        prompt = f"""
        Analyze the following revenue forecast results and provide insights:
        
        Forecasted Results:
        - Total predicted revenue: ${total_predicted:,.2f}
        - Number of campaigns: {num_campaigns}
        - Campaign-level predictions: {scenario_preds}
        
        {f"- Baseline comparison: {context.get('predicted_vs_baseline', 0):.1f}% change" if baseline_period_data else ""}
        
        Please provide:
        1. A brief executive summary of the forecast
        2. Key insights about campaign performance expectations
        3. Specific recommendations for budget optimization
        4. Any risks or considerations to keep in mind
        
        Format as a clear, actionable business summary.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert digital marketing analyst specializing in budget optimization and revenue forecasting."},
            {"role": "user", "content": prompt}
        ]
        
        summary = call_chat(messages, max_tokens=500, temperature=0.7)
        
        if summary:
            logger.info("Generated AI forecast summary")
            return summary
        else:
            # Fallback to basic summary
            return _generate_basic_forecast_summary(scenario_preds, baseline_period_data)
        
    except Exception as e:
        logger.error(f"Error generating forecast summary: {e}")
        return _generate_basic_forecast_summary(scenario_preds, baseline_period_data)

def _generate_basic_forecast_summary(scenario_preds, baseline_period_data=None):
    """Generate a basic forecast summary without AI."""
    if not scenario_preds:
        return "No forecast data available."
    
    total_predicted = sum(scenario_preds.values())
    num_campaigns = len(scenario_preds)
    
    # Find top performers
    sorted_campaigns = sorted(scenario_preds.items(), key=lambda x: x[1], reverse=True)
    top_campaign = sorted_campaigns[0]
    
    summary = f"""
    📊 REVENUE FORECAST SUMMARY
    
    Total Predicted Revenue: ${total_predicted:,.2f}
    Campaigns Analyzed: {num_campaigns}
    
    🎯 TOP PERFORMER
    Campaign {top_campaign[0]}: ${top_campaign[1]:,.2f}
    
    💡 KEY INSIGHTS
    - Average revenue per campaign: ${total_predicted/num_campaigns:,.2f}
    """
    
    if baseline_period_data:
        baseline_total = baseline_period_data.get('total_revenue', 0)
        if baseline_total > 0:
            change_pct = (total_predicted - baseline_total) / baseline_total * 100
            summary += f"- Projected change vs baseline: {change_pct:+.1f}%\n"
    
    summary += "\n📈 RECOMMENDATIONS\n- Review top-performing campaigns for scaling opportunities\n- Consider reallocating budget from underperformers\n- Monitor actual performance against forecasts"
    
    return summary

def create_budget_scenarios(current_spend_data, scenario_types=['conservative', 'aggressive', 'rebalanced']):
    """
    Create different budget scenarios for what-if analysis.
    
    Args:
        current_spend_data: dict {campaign_id: current_spend}
        scenario_types: list of scenario types to generate
    
    Returns:
        dict {scenario_name: {campaign_id: new_spend}}
    """
    if not current_spend_data:
        return {}
    
    scenarios = {}
    total_current_spend = sum(current_spend_data.values())
    
    try:
        if 'conservative' in scenario_types:
            # 10% increase across all campaigns
            scenarios['conservative'] = {
                campaign_id: spend * 1.1 
                for campaign_id, spend in current_spend_data.items()
            }
        
        if 'aggressive' in scenario_types:
            # 50% increase across all campaigns
            scenarios['aggressive'] = {
                campaign_id: spend * 1.5 
                for campaign_id, spend in current_spend_data.items()
            }
        
        if 'rebalanced' in scenario_types:
            # Keep total spend same but reallocate to top performers
            # This is a simplified approach - in practice, would use performance data
            campaign_spends = list(current_spend_data.values())
            avg_spend = np.mean(campaign_spends)
            
            scenarios['rebalanced'] = {}
            for campaign_id, current_spend in current_spend_data.items():
                if current_spend > avg_spend:
                    # Boost above-average campaigns
                    scenarios['rebalanced'][campaign_id] = current_spend * 1.2
                else:
                    # Reduce below-average campaigns
                    scenarios['rebalanced'][campaign_id] = current_spend * 0.8
        
        logger.info(f"Generated {len(scenarios)} budget scenarios")
        return scenarios
        
    except Exception as e:
        logger.error(f"Error creating budget scenarios: {e}")
        return {}

def analyze_roas_forecast(forecast_results, scenario_spend):
    """
    Calculate forecasted ROAS for different scenarios.
    
    Args:
        forecast_results: dict {scenario_name: {campaign_id: predicted_revenue}}
        scenario_spend: dict {scenario_name: {campaign_id: spend}}
    
    Returns:
        DataFrame with ROAS analysis
    """
    try:
        analysis_data = []
        
        for scenario_name in forecast_results.keys():
            if scenario_name not in scenario_spend:
                continue
            
            revenues = forecast_results[scenario_name]
            spends = scenario_spend[scenario_name]
            
            total_revenue = sum(revenues.values())
            total_spend = sum(spends.values())
            total_roas = total_revenue / total_spend if total_spend > 0 else 0
            
            analysis_data.append({
                'scenario': scenario_name,
                'total_spend': total_spend,
                'total_revenue': total_revenue,
                'total_roas': total_roas,
                'num_campaigns': len(revenues)
            })
        
        df = pd.DataFrame(analysis_data)
        df = df.sort_values('total_roas', ascending=False)
        
        logger.info(f"Analyzed ROAS for {len(df)} scenarios")
        return df
        
    except Exception as e:
        logger.error(f"Error analyzing ROAS forecast: {e}")
        return pd.DataFrame()
