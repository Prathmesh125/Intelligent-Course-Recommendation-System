"""Quick test for query understanding"""
from query_engine import understand_query

def main():
    test_queries = [
        "i want to learn python",
        "python",
        "teach me machine learning",
        "chemistry course",
        "how do i learn blockchain",
        "i wanna learn react",
    ]

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
