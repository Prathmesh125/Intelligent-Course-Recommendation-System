"""Quick test for live search"""
from live_search import search_courses_live

test_queries = [
    "i want to learn python",
    "python",
    "machine learning",
]

print("=" * 80)
print("Live Search Test")
print("=" * 80)

for q in test_queries:
    print(f"\nSearching: '{q}'")
    try:
        results, info = search_courses_live(q, top_n=5)
        print(f"✓ Found {len(results)} courses")
        print(f"  Topic: {info['topic']}")
        if results:
            print(f"  First course: {results[0]['course_title'][:60]}...")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    print("-" * 80)
