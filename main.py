
#!/usr/bin/env python3
"""
Main entry point for the AI-powered social campaign optimizer.
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Import and run the dashboard
    from dashboard import main
    main()
