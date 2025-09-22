#!/usr/bin/env python3
"""
Quick API Test Script for Ethical Eye
Tests the trained model API before extension testing
"""

import requests
import json
import time

def test_api():
    """Test the Ethical Eye API with sample texts"""
    
    print("🧪 ETHICAL EYE API TEST")
    print("=" * 50)
    
    # API endpoint
    url = "http://127.0.0.1:5000/analyze"
    
    # Test cases with known dark patterns
    test_cases = [
        {
            "name": "Urgency Pattern",
            "text": "Hurry! Only 2 left in stock! Order now before it's too late!",
            "expected": "Urgency"
        },
        {
            "name": "Scarcity Pattern", 
            "text": "Limited time offer - expires in 24 hours! Don't miss out!",
            "expected": "Scarcity"
        },
        {
            "name": "Social Proof Pattern",
            "text": "Join 10,000+ satisfied customers who love our product!",
            "expected": "Social Proof"
        },
        {
            "name": "Misdirection Pattern",
            "text": "Click here for free shipping" + " (but actually leads to paid options)",
            "expected": "Misdirection"
        },
        {
            "name": "Normal Content",
            "text": "Welcome to our website. We provide excellent customer service.",
            "expected": "Not Dark Pattern"
        }
    ]
    
    print("Testing API with sample dark patterns...")
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"Text: {test_case['text'][:50]}...")
        
        # Prepare request data
        data = {
            "segments": [
                {
                    "text": test_case['text'],
                    "element_id": f"test_{i}",
                    "position": {"x": 0, "y": 0}
                }
            ],
            "confidence_threshold": 0.15  # Lower threshold to catch more patterns
        }
        
        try:
            # Make API request
            start_time = time.time()
            response = requests.post(url, json=data, timeout=10)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                
                if 'results' in result and len(result['results']) > 0:
                    detection = result['results'][0]
                    
                    print(f"✅ Detected: {detection.get('category', 'Unknown')}")
                    print(f"✅ Confidence: {detection.get('confidence', 0):.3f}")
                    print(f"✅ Is Dark Pattern: {detection.get('is_dark_pattern', False)}")
                    
                    if 'explanation' in detection and detection['explanation']:
                        print(f"✅ Explanation: {detection['explanation'][:100]}...")
                    else:
                        print("ℹ️  No explanation available")
                    
                    # Check if detection matches expectation
                    detected_category = detection.get('category', '')
                    if detected_category == test_case['expected']:
                        print("🎯 CORRECT DETECTION!")
                    else:
                        print(f"⚠️  Expected: {test_case['expected']}, Got: {detected_category}")
                    
                    print(f"⏱️  Response Time: {(end_time - start_time)*1000:.1f}ms")
                    
                else:
                    print("❌ No results returned")
                    
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Is the API server running?")
            print("   Run: python api/ethical_eye_api.py")
            
        except requests.exceptions.Timeout:
            print("❌ Timeout Error: API took too long to respond")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 50)
    
    print()
    print("🎯 API Test Summary:")
    print("If you see mostly ✅ marks, your API is working correctly!")
    print("If you see ❌ marks, check the API server and model files.")
    print()
    print("Next steps:")
    print("1. Start API server: python api/ethical_eye_api.py")
    print("2. Load Chrome extension from the 'app' folder")
    print("3. Test on real websites with dark patterns")

if __name__ == "__main__":
    test_api()
