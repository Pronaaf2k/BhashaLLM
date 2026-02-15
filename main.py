
import uvicorn
import os
import sys
from pathlib import Path

# Add project root to sys.path explicitly to ensure bhasha package is found
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting BhashaLLM API on port {port}...")
    # Use reload=True for development
    uvicorn.run("bhasha.app.api:app", host='0.0.0.0', port=port, reload=True)
