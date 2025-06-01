#!/usr/bin/env python3
"""
Test script to verify API key validation works with Firebase
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.auth.auth import validate_api_key, check_rate_limit, generate_api_key

def test_api_key_validation():
    """Test API key validation with Firebase."""
    
    print("Testing API Key Validation System")
    print("=" * 50)
    
    # Test 1: Invalid key
    print("\n1. Testing invalid API key...")
    invalid_key = "invalid_key_12345"
    is_valid = validate_api_key(invalid_key)
    print(f"   Key: {invalid_key}")
    print(f"   Valid: {is_valid}")
    assert not is_valid, "Invalid key should return False"
    print("   ✓ Invalid key correctly rejected")
    
    # Test 2: Generate and validate a new key
    print("\n2. Testing key generation and validation...")
    try:
        new_key = generate_api_key("test_user_123", "free")
        print(f"   Generated key: {new_key}")
        
        # Validate the newly generated key
        is_valid = validate_api_key(new_key)
        print(f"   Valid: {is_valid}")
        assert is_valid, "Newly generated key should be valid"
        print("   ✓ Generated key is valid")
        
        # Test rate limiting
        print("\n3. Testing rate limiting...")
        can_proceed = check_rate_limit(new_key)
        print(f"   Rate limit check: {can_proceed}")
        print("   ✓ Rate limit check completed")
        
    except Exception as e:
        print(f"   Error during key generation: {e}")
        print("   Note: This might be expected if Firebase is not properly configured")
    
    # Test 3: Test with a known valid key (if you have one)
    print("\n4. Testing with existing key from dashboard...")
    # Replace this with an actual key from your dashboard
    dashboard_key = input("Enter an API key from your dashboard (or press Enter to skip): ").strip()
    
    if dashboard_key:
        is_valid = validate_api_key(dashboard_key)
        print(f"   Key: {dashboard_key[:20]}...")
        print(f"   Valid: {is_valid}")
        
        if is_valid:
            print("   ✓ Dashboard key is valid!")
            
            # Test rate limiting
            can_proceed = check_rate_limit(dashboard_key)
            print(f"   Rate limit check: {can_proceed}")
        else:
            print("   ✗ Dashboard key is not valid")
            print("   This could indicate a problem with the validation logic")
    else:
        print("   Skipped - no key provided")
    
    print("\n" + "=" * 50)
    print("Testing completed!")
    print("\nNext steps:")
    print("1. Generate an API key in your dashboard at cinder.digital")
    print("2. Use that key in a Python script with ModelDebugger")
    print("3. Verify the validation works end-to-end")

if __name__ == "__main__":
    test_api_key_validation()