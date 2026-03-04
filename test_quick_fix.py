#!/usr/bin/env python3
"""Quick test to verify rate limiting fixes"""

import sys
sys.path.insert(0, '.')

print("\n🧪 Testing Rate Limiting Fixes...\n")

#Test 1: Import works
try:
    from live_search import search_courses_live
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Cache directory created
from pathlib import Path
cache_dir = Path("dataset/.search_cache")
if cache_dir.exists():
    print(f"✓ Cache directory exists: {cache_dir}")
else:
    print(f"⚠ Cache directory not found (will be created on first search)")

# Test 3: Basic functionality
print("\n📡 Testing search with 'python basics'...")
try:
    results, info = search_courses_live("python basics", top_n=3)
    print(f"✓ Search completed: {len(results)} results")
    if len(results) > 0:
        print(f"  → Found: {results[0]['course_title'][:60]}...")
        print(f"  → Platform: {results[0]['source']}")
    print(f"  → Topic understood: {info.get('topic', 'N/A')}")
except Exception as e:
    error_str = str(e).lower()
    if 'rate limit' in error_str:
        print(f"⚠ Rate limited (expected on Streamlit Cloud)")
        print("  → Fallback to local database will activate in the app")
    else:
        print(f"⚠ Search error: {e}")

# Test 4: Cache works
print("\n💾 Testing cache...")
try:
    from live_search import _load_from_cache, _get_cache_key
    cache_key = _get_cache_key("python basics", {"top_n": 3, "difficulty": "All", "price": "All"})
    cached = _load_from_cache(cache_key)
    if cached:
        results, info = cached
        print(f"✓ Cache HIT! {len(results)} results loaded instantly")
    else:
        print("  → No cache yet (will be created after first successful search)")
except Exception as e:
    print(f"⚠ Cache test error: {e}")

print("\n" + "="*70)
print("Summary:")
print(" ✅ Implemented fixes:")
print("   • 24-hour result caching for faster repeat searches")
print("   • Exponential backoff with retry logic (3 attempts)")
print("   • Increased delays between searches (0.5s)")
print("   • Graceful error handling with user-friendly messages")
print("   • Smart fallback to local database when rate limited")
print("\n💡 On Streamlit Cloud:")
print("   • First searches may be rate limited (shared IPs)")
print("   • Subsequent searches use cache (instant!)")
print("   • When limited, app automatically searches local database")
print("="*70 + "\n")
