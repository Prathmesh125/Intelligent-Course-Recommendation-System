"""Quick test for live search"""
import argparse
from live_search import search_courses_live

def main():
    """
    Runs the live search test.
    Accepts an optional custom query from command line; otherwise runs default test queries.
    """
    parser = argparse.ArgumentParser(description="Test live search functionality.")
    parser.add_argument("-q", "--query", type=str, help="Custom query to test")
    parser.add_argument("-n", "--top_n", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()

    test_queries = [
        "i want to learn python",
        "python",
        "machine learning",
    ]

    if args.query:
        test_queries = [args.query]

    print("=" * 80)
    print("Live Search Test")
    print("=" * 80)

    for q in test_queries:
        print(f"\nSearching: '{q}' (top {args.top_n})")
        try:
            results, info = search_courses_live(q, top_n=args.top_n)
            print(f"✓ Found {len(results)} courses")
            print(f"  Topic: {info.get('topic', 'N/A')}")
            if results:
                print(f"  First course: {results[0]['course_title'][:60]}...")
        except Exception as e:
            print(f"✗ Error: {str(e)}")
        print("-" * 80)

if __name__ == "__main__":
    main()
