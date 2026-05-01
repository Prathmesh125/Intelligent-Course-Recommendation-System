"""Quick test for query understanding"""
import argparse
from query_engine import understand_query

def main():
    """
    Runs the query understanding test. 
    Accepts an optional custom query from command line; otherwise runs default test queries.
    """
    parser = argparse.ArgumentParser(description="Test query understanding.")
    parser.add_argument("-q", "--query", type=str, help="Custom query to test")
    args = parser.parse_args()

    test_queries = [
        "i want to learn python",
        "python",
        "teach me machine learning",
        "chemistry course",
        "how do i learn blockchain",
        "i wanna learn react",
    ]

    if args.query:
        test_queries = [args.query]

    print("=" * 80)
    print("Query Understanding Test")
    print("=" * 80)

    for q in test_queries:
        result = understand_query(q)
        print(f"\nINPUT:   '{q}'")
        print(f"TOPIC:   '{result.get('topic', 'N/A')}'")
        print(f"LEVEL:   {result.get('difficulty', 'N/A')}")
        search_queries = result.get('search_queries', ['N/A'])
        print(f"SEARCH:  {search_queries[0] if search_queries else 'N/A'}")
        print("-" * 80)

if __name__ == "__main__":
    main()
