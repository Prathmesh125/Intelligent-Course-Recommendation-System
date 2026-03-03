"""Quick test for query understanding"""
from query_engine import understand_query

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
    print(f"TOPIC:   '{result['topic']}'")
    print(f"LEVEL:   {result['difficulty']}")
    print(f"SEARCH:  {result['search_queries'][0]}")
    print("-" * 80)
