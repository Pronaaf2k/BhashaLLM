import requests
import re

url = "https://shahariarrabby.github.io/ekush"
try:
    response = requests.get(url, timeout=10)
    print(f"Status: {response.status_code}")
    
    # Simple regex to find hrefs
    links = re.findall(r'href=[\'"]?([^\'" >]+)', response.text)
    
    print("Found links:")
    for link in links:
        if "drive.google.com" in link or "kaggle.com" in link or "download" in link.lower() or ".zip" in link:
            print(link)
            
except Exception as e:
    print(f"Error: {e}")
