"""Clean startup script for the FastAPI server with suppressed TensorFlow warnings."""
import os
import sys

# Suppress TensorFlow warnings before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (0=all, 1=info, 2=warning, 3=error)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU only

# Suppress Python warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Now import and run the app
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Fake News Detection API...")
    print(f"📍 Server will be available at: http://127.0.0.1:{8000}")
    print(f"📚 API Documentation: http://127.0.0.1:{8000}/docs")
    print("=" * 60)
    
    uvicorn.run(
        "app.main:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info",
        access_log=False  # Disable access logs for cleaner output
    )
