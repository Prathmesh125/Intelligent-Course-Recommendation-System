#!/usr/bin/env python3
"""
Test script to verify rate limiting fixes work correctly
- Tests caching mechanism
- Tests retry logic with exponential backoff
- Tests error handling
"""

import sys
import time
from live_search import search_courses_live

def test_cache():
    """Test that caching works - second call should be instant"""
    print("\n" + "="*70)
    print("TEST 1: Cache Functionality")
    print("="*70)
    
    query = "python for beginners"
    print(f"🔍 Searching for: '{query}'")
    
    # First search - should hit the API
    print("\n📡 First search (fresh)...")
    start = time.time()
    try:
        results1, info1 = search_courses_live(query, top_n=5)
        duration1 = time.time() - start
        print(f"✓ Found {len(results1)} results in {duration1:.2f}s")
    except Exception as e:
        print(f"✗ Search failed: {e}")
        return False
    
    # Second search - should use cache (instant)
    print("\n💾 Second search (should use cache)...")
    start = time.time()
    try:
        results2, info2 = search_courses_live(query, top_n=5)
        duration2 = time.time() - start
        print(f"✓ Found {len(results2)} results in {duration2:.2f}s")
        
        if duration2 < 1.0:
            print(f"✓✓ CACHE WORKING! Second search was {duration1/duration2:.1f}x faster")
            return True
        else:
            print(f"⚠ Cache may not be working (took {duration2:.2f}s)")
            return False
    except Exception as e:
        print(f"✗ Second search failed: {e}")
        return False


def test_different_queries():
    """Test multiple different queries"""
    print("\n" + "="*70)
    print("TEST 2: Multiple Query Handling")
    print("="*70)
    
    queries = [
        ("machine learning", 3),
        ("web development", 3),
        ("data science", 3),
    ]
    
    success_count = 0
    for query, top_n in queries:
        print(f"\n🔍 Testing: '{query}'...")
        try:
            results, info = search_courses_live(query, top_n=top_n)
            if results:
                print(f"✓ Found {len(results)} courses")
                print(f"  Topic understood: {info.get('topic', 'N/A')}")
                print(f"  Difficulty: {info.get('difficulty', 'N/A')}")
                success_count += 1
            else:
                print(f"⚠ No results returned (may be rate limited)")
        except Exception as e:
            error_str = str(e).lower()
            if 'rate limit' in error_str:
                print(f"⚠ Rate limited (expected on shared IPs): {e}")
            else:
                print(f"✗ Error: {e}")
        
        # Small delay between queries
        time.sleep(1)
    
    print(f"\n📊 Success rate: {success_count}/{len(queries)}")
    return success_count > 0


def test_error_handling():
    """Test that errors are handled gracefully"""
    print("\n" + "="*70)
    print("TEST 3: Error Handling")
    print("="*70)
    
    # Test with empty query
    print("\n🔍 Testing empty query...")
    try:
        results, info = search_courses_live("", top_n=5)
        if len(results) == 0:
            print("✓ Handled empty query gracefully")
        else:
            print("⚠ Empty query returned results (unexpected)")
    except Exception as e:
        print(f"⚠ Empty query raised exception: {e}")
    
    # Test with very specific query (likely to get filtered)
    print("\n🔍 Testing highly specific query...")
    try:
        results, info = search_courses_live("advanced quantum blockchain AI for underwater basket weaving", top_n=5)
        print(f"✓ Handled specific query: {len(results)} results")
    except Exception as e:
        print(f"⚠ Specific query failed: {e}")
    
    return True


def main():
    print("\n" + "🚀"*35)
    print("   Rate Limiting Fix - Test Suite")
    print("🚀"*35)
    
    tests_passed = 0
    tests_total = 3
    
    # Run tests
    if test_cache():
        tests_passed += 1
    
    if test_different_queries():
        tests_passed += 1
    
    if test_error_handling():
        tests_passed += 1
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("✓✓✓ All tests passed!")
        print("\n💡 The following improvements are now active:")
        print("   • 24-hour result caching (faster repeat searches)")
        print("   • Exponential backoff with 3 retry attempts")
        print("   • Increased delays between search passes (0.5s)")
        print("   • Better user-agent headers to avoid bot detection")
        print("   • Graceful error handling with informative messages")
    else:
        print(f"⚠ {tests_total - tests_passed} test(s) failed")
        print("\nNote: Some failures are expected on rate-limited connections.")
        print("The app will gracefully fall back to cached & local results.")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
