# AI-Powered Social Campaign Optimizer

An intelligent dashboard for optimizing Facebook/Meta advertising campaigns and organic social media performance using AI insights and automated actions.

## 🚀 Features

- **Paid Campaign Analytics**: Fetch and analyze Facebook Ads performance data
- **Organic Content Insights**: Track Facebook Page and Instagram metrics  
- **AI-Powered Anomaly Detection**: Automatically identify performance outliers
- **Automated Optimization**: Smart budget adjustments and bid optimizations
- **Real-time Monitoring**: Live campaign performance tracking
- **Comprehensive Reporting**: Interactive dashboards with Plotly visualizations

## 📋 Prerequisites

- Replit account with Python 3.12 support
- Facebook Business Manager account
- Facebook App with Marketing API access
- Valid Facebook Ad Account
- Facebook Page (for organic insights)

## ⚙️ Setup Instructions

### 1. Configure Replit Secrets

Set the following environment variables in Replit Secrets (🔒 icon in sidebar):

**Required:**
- `META_ACCESS_TOKEN`: Your Facebook Marketing API access token
- `AD_ACCOUNT_ID`: Your Facebook Ad Account ID (numbers only, without "act_" prefix)

**Optional but recommended:**
- `META_APP_ID`: Your Facebook App ID (enables app secret proof for security)
- `META_APP_SECRET`: Your Facebook App Secret  
- `PAGE_ID`: Your Facebook Page ID (for organic insights)

**How to get these values:**
- **Access Token**: [Facebook Business Settings](https://business.facebook.com/settings/system-users) → System Users → Generate Token
- **Ad Account ID**: [Facebook Ads Manager](https://adsmanager.facebook.com) → Account Settings → Account ID
- **App ID/Secret**: [Facebook Developers](https://developers.facebook.com/apps/) → Your App → Basic Settings
- **Page ID**: [Facebook Page](https://www.facebook.com/your-page) → About → Page ID

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify SDK Installation

Run the test script to ensure everything is properly configured:

```bash
python test_sdk_import.py
```

This will check:
- ✅ Facebook Business SDK imports
- ✅ Environment variables  
- ✅ API initialization
- ✅ No local module shadowing

### 4. Run the Application

```bash
streamlit run dashboard.py
```

The app will be available at `https://your-repl-name.replit.app`

## 🔧 Configuration Files

### `.streamlit/config.toml`
Configures Streamlit for headless operation in Replit:
```toml
[server]
headless = true          # prevents browser opening (fixes Python 3.12 distutils error)
enableCORS = false       # avoid CORS issues in Replit
port = 5000
address = "0.0.0.0"
```

### `requirements.txt`
Key dependencies:
- `setuptools-distutils`: Python 3.12 compatibility shim
- `facebook-business==18.0.0`: Facebook Marketing API SDK
- `streamlit`: Web dashboard framework
- `pandas`, `plotly`: Data analysis and visualization

## 🛠️ Troubleshooting

### Common Issues

#### 1. `ModuleNotFoundError: No module named 'distutils'`
**Cause**: Python 3.12 removed `distutils`, but Streamlit tries to import it for browser opening.

**Solutions**:
- ✅ Install `setuptools-distutils`: `pip install setuptools-distutils`
- ✅ Set `headless = true` in `.streamlit/config.toml` 
- ✅ Use the provided configuration files

#### 2. `ImportError: cannot import name 'FacebookAdsApi'` or circular import errors
**Cause**: Local `facebook_business` directory shadowing the installed SDK.

**Solutions**:
```bash
# Remove any local facebook_business directories
find . -name "facebook_business" -type d
rm -rf facebook_business/

# Reinstall the SDK
pip uninstall facebook-business
pip install facebook-business==18.0.0
```

#### 3. `TypeError: catching classes that do not inherit from BaseException`
**Cause**: Incorrectly imported exception classes.

**Solution**: Use only properly imported exceptions:
```python
from facebook_business.exceptions import FacebookRequestError
```

#### 4. Missing Environment Variables
**Error**: `Missing required environment variables: ['META_ACCESS_TOKEN']`

**Solution**: 
1. Go to Replit Secrets (🔒 icon)
2. Add required variables listed above
3. Restart the application

#### 5. Invalid OAuth Token (Error Code 190)
**Cause**: Expired or invalid access token.

**Solution**:
- Generate new long-lived token in Facebook Business Settings
- Update `META_ACCESS_TOKEN` in Replit Secrets
- Long-lived tokens expire after ~60 days

#### 6. Permission Errors
**Cause**: App doesn't have required Marketing API permissions.

**Solution**:
- Ensure your Facebook App has `ads_read` permission
- Submit for review if needed for live data access
- Use test ad accounts during development

### Debugging Commands

```bash
# Check for local shadow modules
find . -name "facebook_business"

# Verify SDK installation
python -c "import facebook_business; print(facebook_business.__file__)"

# Test imports and initialization  
python test_sdk_import.py

# Check environment variables (without exposing values)
python -c "import os; print({k: bool(v) for k, v in os.environ.items() if k.startswith('META_') or k == 'AD_ACCOUNT_ID'})"
```

### Log Analysis

Check console output for these success indicators:
- `✅ facebook_business.api loaded from: /path/to/installed/package`
- `✅ Facebook SDK initialized for Ad Account: act_XXXXXXXXX`
- `✅ Facebook API Connected` (in sidebar)

## 📚 Official Documentation References

- **Facebook Business SDK**: https://developers.facebook.com/docs/business-sdk/
- **Marketing API Insights**: https://developers.facebook.com/docs/marketing-api/insights/
- **Page Insights API**: https://developers.facebook.com/docs/graph-api/reference/page/insights/
- **Streamlit Configuration**: https://docs.streamlit.io/library/advanced-features/configuration
- **Python 3.12 Changes**: https://docs.python.org/3/whatsnew/3.12.html

## 🔄 Data Flow

1. **Environment Check**: Validate API credentials and SDK imports
2. **API Initialization**: Connect to Facebook Marketing API with app secret proof
3. **Data Fetching**: Retrieve campaign performance and organic insights
4. **Processing**: Clean, normalize, and store data in local SQLite database
5. **Analysis**: Run anomaly detection and generate AI insights
6. **Visualization**: Display interactive dashboards with real-time updates
7. **Automation**: Execute approved optimizations with safety checks

## 🛡️ Security Best Practices

- Store all sensitive credentials in Replit Secrets (never in code)
- Use app secret proof for enhanced API security
- Implement rate limiting for API calls
- Log all automated actions for audit trails
- Test automations in dry-run mode first

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Update documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

**Need Help?** 
- Check the troubleshooting section above
- Run `python test_sdk_import.py` for diagnostic information
- Review console logs for specific error details
- Consult official Facebook API documentation