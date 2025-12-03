#!/usr/bin/env python3
"""
SHAP Verification Test Script
Tests if SHAP/gradient-based importance is working correctly
"""

import requests
import json

def test_shap():
    """Test if SHAP values are being computed and returned"""
    
    print("SHAP VERIFICATION TEST")
    print("=" * 60)
    print()
    
    # Test data with known dark patterns
    test_cases = [
        {
            "name": "Urgency Pattern",
            "text": "Hurry! Only 2 left in stock! Order now before it's too late!",
        },
        {
            "name": "Scarcity Pattern", 
            "text": "Limited time offer - expires in 24 hours! Don't miss out!",
        },
        {
            "name": "Social Proof Pattern",
            "text": "Join 10,000+ satisfied customers who love our product!",
        }
    ]
    
    url = "http://127.0.0.1:5000/analyze_single"
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"Text: \"{test_case['text']}\"")
        print("-" * 60)
        
        try:
            data = {
                "text": test_case['text'],
                "confidence_threshold": 0.5
            }
            
            response = requests.post(url, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                # Check basic fields
                print(f"[OK] Category: {result.get('category', 'N/A')}")
                print(f"[OK] Confidence: {result.get('confidence', 0):.3f}")
                print(f"[OK] Is Dark Pattern: {result.get('is_dark_pattern', False)}")
                print()
                
                # Check SHAP-specific fields
                print("SHAP Data Check:")
                print("-" * 60)
                
                # Check top_words
                top_words = result.get('top_words', [])
                if top_words:
                    print(f"[OK] top_words: {len(top_words)} words found")
                    print("   Top contributing words:")
                    for word, score in top_words[:5]:
                        print(f"      - '{word}': {score:.4f}")
                else:
                    print("[FAIL] top_words: EMPTY or MISSING")
                
                print()
                
                # Check shap_values
                shap_values = result.get('shap_values', [])
                if shap_values:
                    print(f"[OK] shap_values: {len(shap_values)} values found")
                    print(f"   Min: {min(shap_values):.4f}, Max: {max(shap_values):.4f}, Avg: {sum(shap_values)/len(shap_values):.4f}")
                    # Show non-zero values
                    non_zero = [v for v in shap_values if abs(v) > 0.001]
                    print(f"   Non-zero values: {len(non_zero)}/{len(shap_values)}")
                else:
                    print("[FAIL] shap_values: EMPTY or MISSING")
                
                print()
                
                # Check tokens
                tokens = result.get('tokens', [])
                if tokens:
                    print(f"[OK] tokens: {len(tokens)} tokens found")
                    print(f"   Sample tokens: {tokens[:10]}")
                else:
                    print("[FAIL] tokens: EMPTY or MISSING")
                
                print()
                
                # Check explanation
                explanation = result.get('explanation', '')
                if explanation:
                    print(f"[OK] explanation: Present")
                    print(f"   {explanation[:150]}...")
                else:
                    print("[FAIL] explanation: MISSING")
                
                print()
                
                # Overall SHAP status
                has_shap_data = bool(top_words and shap_values and tokens)
                if has_shap_data:
                    print("*** SHAP IS WORKING! ***")
                    print("   - top_words: [OK]")
                    print("   - shap_values: [OK]")
                    print("   - tokens: [OK]")
                else:
                    print("*** SHAP DATA INCOMPLETE ***")
                    if not top_words:
                        print("   - top_words: [FAIL]")
                    if not shap_values:
                        print("   - shap_values: [FAIL]")
                    if not tokens:
                        print("   - tokens: [FAIL]")
                
            else:
                print(f"[FAIL] API Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("[FAIL] Cannot connect to API. Is the server running?")
            print("   Run: python api/ethical_eye_api.py")
            break
        except Exception as e:
            print(f"[FAIL] Error: {e}")
        
        print()
        print("=" * 60)
        print()
    
    print("SUMMARY")
    print("=" * 60)
    print("To verify SHAP is working, check:")
    print("1. [OK] top_words should contain (word, score) tuples")
    print("2. [OK] shap_values should be a list of numbers")
    print("3. [OK] tokens should be a list of token strings")
    print("4. [OK] All three should be non-empty for dark patterns")
    print()
    print("If any are missing/empty, SHAP computation may have failed.")
    print("Check the API server logs for error messages.")

if __name__ == "__main__":
    test_shap()

