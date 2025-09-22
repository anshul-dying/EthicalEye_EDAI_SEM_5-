#!/usr/bin/env python3
"""
Quick API Test - Simple version
"""

import requests
import json

def quick_test():
    """Quick test of the API"""
    
    print("🧪 QUICK API TEST")
    print("=" * 30)
    
    # Test data
    data = {
        "segments": [
            {
                "text": "Hurry! Only 2 left in stock!",
                "element_id": "test_1",
                "position": {"x": 0, "y": 0}
            }
        ],
        "confidence_threshold": 0.15
    }
    
    try:
        response = requests.post("http://127.0.0.1:5000/analyze", json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API is working!")
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
        print("Run: python api/ethical_eye_api.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    quick_test()
