
#!/usr/bin/env python3
"""
Start both the Streamlit dashboard and the API server for SauceRoom integration
"""

import subprocess
import os
import sys
import time
import threading
import signal
from multiprocessing import Process

def start_streamlit():
    """Start the Streamlit dashboard"""
    cmd = [
        sys.executable, "-m", "streamlit", "run", "dashboard.py",
        "--server.port", "5000",
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ]
    
    print("🚀 Starting Streamlit dashboard on port 5000...")
    return subprocess.Popen(cmd)

def start_api_server():
    """Start the API server"""
    cmd = [sys.executable, "api_endpoints.py"]
    
    print("🔗 Starting API server on port 5001...")
    return subprocess.Popen(cmd)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutting down servers...")
    sys.exit(0)

def main():
    """Start both services"""
    print("🌟 Starting AI-Powered Social Campaign Optimizer with SauceRoom Integration")
    print("=" * 70)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    processes = []
    
    try:
        # Start Streamlit dashboard
        streamlit_process = start_streamlit()
        processes.append(streamlit_process)
        
        # Wait a moment
        time.sleep(2)
        
        # Start API server
        api_process = start_api_server()
        processes.append(api_process)
        
        print("\n✅ Both services started successfully!")
        print("📊 Dashboard: https://your-repl-name.replit.app")
        print("🔗 API: https://your-repl-name.replit.app:5001")
        print("📚 API Docs: Check the dashboard for integration details")
        print("\nPress Ctrl+C to stop both services")
        
        # Wait for processes
        for process in processes:
            process.wait()
            
    except KeyboardInterrupt:
        print("\n🛑 Received shutdown signal")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Clean up processes
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        
        print("✅ Shutdown complete")

if __name__ == "__main__":
    main()
